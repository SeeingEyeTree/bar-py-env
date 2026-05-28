--------------------------------------------------------------------------------
-- bar-py-env bridge widget
--
-- Connects to a Python TCP server, sends per-step observations, and executes
-- per-step actions received back. Communicates with length-prefixed JSON
-- messages (4-byte big-endian length, then UTF-8 JSON).
--
-- Configuration is read from `LuaUI/Config/bar_py_env.txt` in the engine's
-- write-dir. Python writes that file before launching the engine.
--------------------------------------------------------------------------------

function widget:GetInfo()
    return {
        name      = "BAR Python Env Bridge",
        desc      = "TCP bridge between Beyond All Reason and a Python RL env.",
        author    = "bar-py-env",
        date      = "2026",
        license   = "MIT",
        layer     = 0,
        enabled   = true,
        handler   = true,
    }
end

--------------------------------------------------------------------------------
-- Defaults (overridden by config file if present)
--------------------------------------------------------------------------------

local HOST = "127.0.0.1"
local PORT = 8765
local STEP_INTERVAL = 30  -- game frames between obs/action exchanges (~1s at 30 fps)
local MAX_STEPS = 10      -- safety cap; Python normally sends 'quit' first
-- Mode determines the widget's loop shape:
--   "live"   -- existing behavior: send obs, recv actions, apply, quit on MAX_STEPS
--   "replay" -- spectate a .sdfz: send obs + commands the engine just applied,
--               no recv from Python, exit when the engine fires GameOver.
local MODE = "live"
-- Target gamespeed multiplier. Engine default is 1x; in replay/headless we
-- usually want to crank this way up so 10 minutes of game time finishes in
-- ~30 seconds of wall clock. The hard cap is the `maxspeed` modoption (BAR
-- typically allows up to 20x; our generated startscripts set 100x).
local GAME_SPEED = 20

-- Cached team IDs, set once in Initialize() to guard against any late-game
-- drift in Spring.GetMyTeamID() / Spring.GetMyAllyTeamID().
local MY_TEAM_ID = nil
local MY_ALLY_ID = nil

-- Metal extraction spots, computed once (lazily on first game frame) and cached.
-- Included in every observation's `map` field so Python agents can plan builds.
local METAL_SPOTS = nil

--------------------------------------------------------------------------------
-- Tiny JSON encoder/decoder (rxi/json.lua, MIT). Inlined to avoid VFS lookups.
-- Source: https://github.com/rxi/json.lua  (slightly trimmed)
--------------------------------------------------------------------------------

local json = {}
do
    local function escape_char(c)
        local ESC = { ['"']='\\"', ['\\']='\\\\', ['\b']='\\b', ['\f']='\\f',
                      ['\n']='\\n', ['\r']='\\r', ['\t']='\\t' }
        return ESC[c] or string.format("\\u%04x", c:byte())
    end
    local function encode_string(s) return '"' .. s:gsub('[%z\1-\31\\"]', escape_char) .. '"' end
    local function encode_number(n)
        if n ~= n or n == math.huge or n == -math.huge then return "null" end
        if n == math.floor(n) and math.abs(n) < 1e16 then return string.format("%d", n) end
        return string.format("%.14g", n)
    end
    local encode
    local function encode_table(t)
        -- array vs object
        local n = 0
        for k in pairs(t) do
            if type(k) ~= "number" then n = -1; break end
            if k > n then n = k end
        end
        local out = {}
        if n > 0 then
            for i = 1, n do out[#out + 1] = encode(t[i]) end
            return "[" .. table.concat(out, ",") .. "]"
        elseif n == 0 then
            return "[]"
        else
            for k, v in pairs(t) do
                out[#out + 1] = encode_string(tostring(k)) .. ":" .. encode(v)
            end
            return "{" .. table.concat(out, ",") .. "}"
        end
    end
    function encode(v)
        local tv = type(v)
        if v == nil then return "null"
        elseif tv == "string" then return encode_string(v)
        elseif tv == "number" then return encode_number(v)
        elseif tv == "boolean" then return v and "true" or "false"
        elseif tv == "table" then return encode_table(v)
        else return "null" end
    end
    json.encode = encode

    -- Decoder
    local pos
    local decode_value
    local function err(s, msg) error(("json decode error at byte %d: %s"):format(pos, msg)) end
    local function skip_ws(s)
        local _, p = s:find("^[ \t\r\n]*", pos); pos = p + 1
    end
    local function decode_string(s)
        local i = pos + 1
        local out = {}
        while i <= #s do
            local c = s:sub(i, i)
            if c == '"' then pos = i + 1; return table.concat(out)
            elseif c == "\\" then
                local n = s:sub(i + 1, i + 1)
                if n == '"' or n == "\\" or n == "/" then out[#out+1] = n; i = i + 2
                elseif n == "b" then out[#out+1] = "\b"; i = i + 2
                elseif n == "f" then out[#out+1] = "\f"; i = i + 2
                elseif n == "n" then out[#out+1] = "\n"; i = i + 2
                elseif n == "r" then out[#out+1] = "\r"; i = i + 2
                elseif n == "t" then out[#out+1] = "\t"; i = i + 2
                elseif n == "u" then
                    local hex = s:sub(i + 2, i + 5)
                    out[#out+1] = string.char(tonumber(hex, 16) % 256)
                    i = i + 6
                else err(s, "bad escape") end
            else
                out[#out+1] = c; i = i + 1
            end
        end
        err(s, "unterminated string")
    end
    local function decode_number(s)
        local a, b = s:find("^-?%d+%.?%d*[eE]?[%+%-]?%d*", pos)
        local n = tonumber(s:sub(a, b))
        pos = b + 1
        return n
    end
    local function decode_literal(s, lit, val)
        if s:sub(pos, pos + #lit - 1) == lit then pos = pos + #lit; return val end
        err(s, "expected " .. lit)
    end
    local function decode_array(s)
        local out = {}; pos = pos + 1; skip_ws(s)
        if s:sub(pos, pos) == "]" then pos = pos + 1; return out end
        while true do
            out[#out + 1] = decode_value(s); skip_ws(s)
            local c = s:sub(pos, pos)
            if c == "," then pos = pos + 1; skip_ws(s)
            elseif c == "]" then pos = pos + 1; return out
            else err(s, "expected , or ]") end
        end
    end
    local function decode_object(s)
        local out = {}; pos = pos + 1; skip_ws(s)
        if s:sub(pos, pos) == "}" then pos = pos + 1; return out end
        while true do
            skip_ws(s)
            if s:sub(pos, pos) ~= '"' then err(s, "expected string key") end
            local k = decode_string(s); skip_ws(s)
            if s:sub(pos, pos) ~= ":" then err(s, "expected :") end
            pos = pos + 1; skip_ws(s)
            out[k] = decode_value(s); skip_ws(s)
            local c = s:sub(pos, pos)
            if c == "," then pos = pos + 1
            elseif c == "}" then pos = pos + 1; return out
            else err(s, "expected , or }") end
        end
    end
    function decode_value(s)
        skip_ws(s)
        local c = s:sub(pos, pos)
        if c == '"' then return decode_string(s)
        elseif c == "{" then return decode_object(s)
        elseif c == "[" then return decode_array(s)
        elseif c == "t" then return decode_literal(s, "true", true)
        elseif c == "f" then return decode_literal(s, "false", false)
        elseif c == "n" then return decode_literal(s, "null", nil)
        else return decode_number(s) end
    end
    function json.decode(s) pos = 1; return decode_value(s) end
end

--------------------------------------------------------------------------------
-- Config loading
--------------------------------------------------------------------------------

local function load_config()
    -- Reads simple `key=value` lines from LuaUI/Config/bar_py_env.txt.
    local txt = VFS.LoadFile("LuaUI/Config/bar_py_env.txt", VFS.RAW)
    if not txt then return end
    for line in txt:gmatch("[^\r\n]+") do
        local k, v = line:match("^%s*(%S+)%s*=%s*(.-)%s*$")
        if k == "host" then HOST = v
        elseif k == "port" then PORT = tonumber(v) or PORT
        elseif k == "step_interval" then STEP_INTERVAL = tonumber(v) or STEP_INTERVAL
        elseif k == "max_steps" then MAX_STEPS = tonumber(v) or MAX_STEPS
        elseif k == "mode" then MODE = v
        elseif k == "speed" then GAME_SPEED = tonumber(v) or GAME_SPEED
        end
    end
end

--------------------------------------------------------------------------------
-- Wire protocol (length-prefixed JSON over TCP)
--------------------------------------------------------------------------------

local function pack_u32_be(n)
    return string.char(
        math.floor(n / 0x1000000) % 256,
        math.floor(n / 0x10000) % 256,
        math.floor(n / 0x100) % 256,
        n % 256
    )
end

local function unpack_u32_be(s)
    local b1, b2, b3, b4 = s:byte(1, 4)
    return b1 * 0x1000000 + b2 * 0x10000 + b3 * 0x100 + b4
end

local function send_msg(sock, msg)
    local body = json.encode(msg)
    local ok, err = sock:send(pack_u32_be(#body) .. body)
    return ok, err
end

local function recv_exact(sock, n)
    local parts = {}
    local got = 0
    while got < n do
        local chunk, err, partial = sock:receive(n - got)
        if chunk then
            parts[#parts + 1] = chunk
            got = got + #chunk
        elseif err == "timeout" then
            if partial and #partial > 0 then
                parts[#parts + 1] = partial
                got = got + #partial
            end
            -- keep looping; with default blocking socket this rarely fires
        else
            return nil, err
        end
    end
    return table.concat(parts)
end

local function recv_msg(sock)
    local hdr, err = recv_exact(sock, 4)
    if not hdr then return nil, err end
    local len = unpack_u32_be(hdr)
    if len <= 0 or len > 64 * 1024 * 1024 then
        return nil, "implausible message length " .. tostring(len)
    end
    local body, err2 = recv_exact(sock, len)
    if not body then return nil, err2 end
    return json.decode(body)
end

--------------------------------------------------------------------------------
-- Observation builder
--------------------------------------------------------------------------------

local function build_observation(frame, done, result)
    local my_team = MY_TEAM_ID or Spring.GetMyTeamID()
    local my_ally = MY_ALLY_ID or Spring.GetMyAllyTeamID()

    local metal, metal_storage, metal_pull, metal_income = Spring.GetTeamResources(my_team, "metal")
    local energy, energy_storage, energy_pull, energy_income = Spring.GetTeamResources(my_team, "energy")

    local my_units = {}
    for _, uid in ipairs(Spring.GetTeamUnits(my_team) or {}) do
        local x, y, z = Spring.GetUnitPosition(uid)
        local hp, max_hp = Spring.GetUnitHealth(uid)
        local def_id = Spring.GetUnitDefID(uid)
        my_units[#my_units + 1] = {
            id = uid, def = def_id,
            x = x or 0, y = y or 0, z = z or 0,
            hp = hp or 0, max_hp = max_hp or 0,
        }
    end

    local enemies = {}
    for _, uid in ipairs(Spring.GetVisibleUnits(my_ally, nil, true) or {}) do
        if Spring.GetUnitAllyTeam(uid) ~= my_ally then
            local x, y, z = Spring.GetUnitPosition(uid)
            local def_id = Spring.GetUnitDefID(uid)
            enemies[#enemies + 1] = {
                id = uid, def = def_id or 0,
                x = x or 0, y = y or 0, z = z or 0,
            }
        end
    end

    return {
        type = "obs",
        frame = frame,
        done = done or false,
        result = result,
        team = my_team,
        ally = my_ally,
        map = { size_x = Game.mapSizeX, size_z = Game.mapSizeZ, metal_spots = METAL_SPOTS },
        resources = {
            metal = metal or 0, metal_storage = metal_storage or 0,
            metal_pull = metal_pull or 0, metal_income = metal_income or 0,
            energy = energy or 0, energy_storage = energy_storage or 0,
            energy_pull = energy_pull or 0, energy_income = energy_income or 0,
        },
        my_units = my_units,
        enemies = enemies,
    }
end

--------------------------------------------------------------------------------
-- Replay-mode observation builder (god's-eye: iterates every team).
-- During replay playback our local player is a spectator with no team, so the
-- live build_observation() sees nothing. Here we serialize all non-Gaia teams.
--------------------------------------------------------------------------------

local function build_observation_replay(frame, done, result, commands_this_frame)
    local gaia      = Spring.GetGaiaTeamID()
    local gaia_ally = Spring.GetTeamAllyTeamID(gaia)

    -- Collect ally team IDs that belong to real players (exclude Gaia's).
    local player_ally_teams = {}
    for _, at in ipairs(Spring.GetAllyTeamList() or {}) do
        if at ~= gaia_ally then
            player_ally_teams[#player_ally_teams + 1] = at
        end
    end

    -- Teams + units -----------------------------------------------------------
    local teams = {}
    for _, team_id in ipairs(Spring.GetTeamList() or {}) do
        if team_id ~= gaia then
            local m, ms, mp, mi = Spring.GetTeamResources(team_id, "metal")
            local e, es, ep, ei = Spring.GetTeamResources(team_id, "energy")
            local unit_ally = Spring.GetTeamAllyTeamID(team_id)
            local units = {}
            for _, uid in ipairs(Spring.GetTeamUnits(team_id) or {}) do
                local x, y, z           = Spring.GetUnitPosition(uid)
                -- GetUnitHealth returns: health, maxHealth, paralyzeDmg, captureProgress, buildProgress
                local hp, max_hp, _, _, bp = Spring.GetUnitHealth(uid)
                -- Heading: Spring value 0-65535; divide by 256 → compact 0-255 (≈1.4° resolution).
                local heading           = Spring.GetUnitHeading(uid)

                local u = {
                    id = uid, def = Spring.GetUnitDefID(uid),
                    x = x or 0, y = y or 0, z = z or 0,
                    hp = hp or 0, max_hp = max_hp or 0,
                    h  = math.floor((heading or 0) / 256),
                }

                if bp and bp > 0 and bp < 1 then u.bp = bp end

                -- Visibility from enemy ally teams.
                -- los   = full line-of-sight (position + type + hp known to that team)
                -- radar = radar detection only (dot on minimap; type/hp unknown)
                local los_list, radar_list = {}, {}
                for _, at in ipairs(player_ally_teams) do
                    if at ~= unit_ally then
                        if Spring.IsUnitInLos(uid, at) then
                            los_list[#los_list + 1] = at
                        elseif Spring.IsUnitOnRadar and Spring.IsUnitOnRadar(uid, at) then
                            radar_list[#radar_list + 1] = at
                        end
                    end
                end
                if #los_list   > 0 then u.los   = los_list   end
                if #radar_list > 0 then u.radar = radar_list end

                units[#units + 1] = u
            end
            teams[tostring(team_id)] = {
                ally = unit_ally,
                metal = m or 0, metal_storage = ms or 0,
                metal_pull = mp or 0, metal_income = mi or 0,
                energy = e or 0, energy_storage = es or 0,
                energy_pull = ep or 0, energy_income = ei or 0,
                units = units,
            }
        end
    end

    -- Features (wrecks, trees, rocks — anything with reclaimable value) -------
    -- GetFeatureResources returns: totalMetal, totalEnergy, reclaimLeft (0-1), reclaimTime
    -- Actual remaining value = total * reclaimLeft.
    local features = {}
    for _, fid in ipairs(Spring.GetAllFeatures() or {}) do
        local tot_m, tot_e, rl = Spring.GetFeatureResources(fid)
        rl = rl or 1
        local cur_m = (tot_m or 0) * rl
        local cur_e = (tot_e or 0) * rl
        if cur_m > 0 or cur_e > 0 then
            local fx, fy, fz = Spring.GetFeaturePosition(fid)
            local f = {
                id  = fid,
                def = Spring.GetFeatureDefID(fid),
                x   = math.floor(fx or 0),
                z   = math.floor(fz or 0),
                m   = cur_m,
            }
            if cur_e > 0 then f.e = cur_e end
            -- Visibility: which player ally teams can see this map position.
            local los_list = {}
            for _, at in ipairs(player_ally_teams) do
                if Spring.IsPosInLos(fx or 0, fy or 0, fz or 0, at) then
                    los_list[#los_list + 1] = at
                end
            end
            if #los_list > 0 then f.los = los_list end
            features[#features + 1] = f
        end
    end
    return {
        type = "obs",
        mode = "replay",
        frame = frame,
        done = done or false,
        result = result,
        map = { size_x = Game.mapSizeX, size_z = Game.mapSizeZ, metal_spots = METAL_SPOTS },
        teams    = teams,
        features = features,
        commands_this_frame = commands_this_frame or {},
    }
end

--------------------------------------------------------------------------------
-- Action application
--
-- An action message from Python looks like:
--   {
--     "unit": <unitID>            -- or "units": [uid1, uid2, ...] for groups
--     "cmd":  <string-or-int>     -- e.g. "MOVE" / "BUILD" / -209
--     "def":  <unitDefID>         -- only for "BUILD"; the unit to build
--     "params": [<numbers>]       -- cmd-specific (see comments below)
--     "options": {                -- optional; default = no modifiers
--       "shift": bool, "ctrl": bool, "alt": bool,
--       "right": bool, "meta": bool, "internal": bool
--     }
--   }
--
-- Common param shapes:
--   MOVE / FIGHT / PATROL: [x, y, z]               (y is usually map height there)
--   ATTACK (unit)        : [targetUnitID]
--   ATTACK (ground)      : [x, y, z]
--   AREA_ATTACK          : [x, y, z, radius]
--   REPAIR / GUARD       : [targetUnitID] OR [x, y, z, radius]
--   RECLAIM (unit)       : [unitID]                (unit-id space)
--   RECLAIM (feature)    : [Game.maxUnits + featureID]  (feature-id space)
--   RECLAIM (area)       : [x, y, z, radius]
--   RESURRECT            : [Game.maxUnits + featureID] OR [x, y, z, radius]
--   BUILD                : [x, y, z, facing]       (facing: 0=S, 1=E, 2=N, 3=W)
--   FIRE_STATE           : [0|1|2]  (hold-fire / return-fire / fire-at-will)
--   MOVE_STATE           : [0|1|2]  (hold-position / maneuver / roam)
--   ONOFF                : [0|1]
--   STOP / WAIT / SELFD  : []
--   STOCKPILE            : []        (queues one missile)
--   CLOAK                : [0|1]
--   TRAJECTORY           : [0|1]
--   DGUN                 : [targetUnitID] OR [x, y, z]
--   LOAD_UNITS           : [unitID] OR [x, y, z, radius]
--   UNLOAD_UNITS         : [x, y, z, radius]
--------------------------------------------------------------------------------

-- Named-command table. Negative values are BUILD commands (cmd_id = -unitDefID);
-- those are constructed dynamically from act.def, not table-mapped.
local CMD_NAME_TO_ID = {
    -- Movement
    STOP        = CMD.STOP,
    WAIT        = CMD.WAIT,
    MOVE        = CMD.MOVE,
    FIGHT       = CMD.FIGHT,
    PATROL      = CMD.PATROL,
    GUARD       = CMD.GUARD,
    -- Combat
    ATTACK      = CMD.ATTACK,
    AREA_ATTACK = CMD.AREA_ATTACK,
    MANUALFIRE  = CMD.MANUALFIRE,    -- commander d-gun
    DGUN        = CMD.MANUALFIRE,    -- alias
    -- Construction / economy
    REPAIR      = CMD.REPAIR,
    RECLAIM     = CMD.RECLAIM,
    RESURRECT   = CMD.RESURRECT,
    CAPTURE     = CMD.CAPTURE,
    RESTORE     = CMD.RESTORE,
    -- States
    FIRE_STATE  = CMD.FIRE_STATE,
    MOVE_STATE  = CMD.MOVE_STATE,
    ONOFF       = CMD.ONOFF,
    CLOAK       = CMD.CLOAK,
    TRAJECTORY  = CMD.TRAJECTORY,
    REPEAT      = CMD.REPEAT,
    -- Transports
    LOAD_UNITS  = CMD.LOAD_UNITS,
    UNLOAD_UNITS = CMD.UNLOAD_UNITS,
    UNLOAD_UNIT = CMD.UNLOAD_UNIT,
    -- Misc
    SELFD       = CMD.SELFD,
    STOCKPILE   = CMD.STOCKPILE,
    SET_WANTED_MAX_SPEED = CMD.SET_WANTED_MAX_SPEED,
    -- Queue editing (rarely needed by agents but useful for completeness)
    INSERT      = CMD.INSERT,
    REMOVE      = CMD.REMOVE,
}

-- Encode the per-command options table into the bitmask the engine expects.
-- Spring.GiveOrderToUnit accepts EITHER a bitmask integer OR a table with
-- boolean fields; we build the table form for clarity.
local function build_options_table(opt)
    if type(opt) ~= "table" then return {} end
    local t = {}
    -- Recognized boolean flags in Spring's command system.
    if opt.shift    then t[#t+1] = "shift"    end
    if opt.ctrl     then t[#t+1] = "ctrl"     end
    if opt.alt      then t[#t+1] = "alt"      end
    if opt.right    then t[#t+1] = "right"    end
    if opt.meta     then t[#t+1] = "meta"     end
    if opt.internal then t[#t+1] = "internal" end
    return t
end

-- Resolve a single action message to its concrete (cmd_id, params) pair.
-- Returns nil if the action is malformed.
local function resolve_command(act)
    local raw = act.cmd
    local cmd_id
    if type(raw) == "number" then
        -- Caller passed a literal cmd_id (e.g. -209 for BUILD <def 209>).
        cmd_id = raw
    elseif type(raw) == "string" then
        if raw == "BUILD" then
            -- BUILD <unitDef>: cmd_id is the negative of the unit-def id.
            -- act.def can be a numeric unitDefID or a string def name (e.g. "cormex").
            local def
            if type(act.def) == "string" then
                local ud = UnitDefNames[act.def]
                if ud then def = ud.id end
            else
                def = tonumber(act.def)
            end
            if not def or def <= 0 then return nil end
            cmd_id = -def
        else
            cmd_id = CMD_NAME_TO_ID[raw]
        end
    end
    if not cmd_id then return nil end
    local params = act.params or {}
    if type(params) ~= "table" then return nil end
    return cmd_id, params
end

local function apply_actions(actions)
    if not actions then return 0 end
    local n = 0
    for _, act in ipairs(actions) do
        local cmd_id, params = resolve_command(act)
        if cmd_id and params then
            local opts = build_options_table(act.options)
            -- Multi-unit form: { units = {uid1, uid2, ...}, ... }
            if type(act.units) == "table" and #act.units > 0 then
                Spring.GiveOrderToUnitArray(act.units, cmd_id, params, opts)
                n = n + 1
            elseif act.unit then
                Spring.GiveOrderToUnit(act.unit, cmd_id, params, opts)
                n = n + 1
            end
        end
    end
    return n
end

--------------------------------------------------------------------------------
-- Main bridge state
--------------------------------------------------------------------------------

-- Note: we deliberately do NOT declare `local socket = nil` here. BAR's user-
-- widget environment exposes the LuaSocket module as a global named `socket`
-- via the `System` metatable (luaui/system.lua). A `local socket` declaration
-- would shadow that global and silently break LuaSocket access.

local sock
local connected = false
local steps_done = 0
local quitting = false
local last_step_frame = -1
-- Quit is a 2-phase sequence so the demo file gets finalized:
--   phase nil   -> not quitting yet
--   phase "go"  -> Spring.GameOver has been called; waiting a few frames for
--                  game_end.lua + the demo writer to flush before forcing exit
--   phase "done"-> "quitforce" has been issued; subsequent ticks no-op
local quit_phase = nil
local quit_phase_until_frame = -1

-- Replay-mode command capture. widget:UnitCommand fires each time the engine
-- applies a command (whether issued by a player, an AI, or replayed from the
-- demo stream). We accumulate them per frame and flush in build_observation.
local command_buffer = {}

local function log(msg)
    Spring.Echo("[bar-py-env] " .. tostring(msg))
end

local function compute_metal_spots()
    local wg_finder = WG and WG["resource_spot_finder"]
    if wg_finder and wg_finder.metalSpotsList then
        local spots = {}
        for _, s in ipairs(wg_finder.metalSpotsList) do
            spots[#spots + 1] = {
                x = math.floor(s.x or 0),
                z = math.floor(s.z or 0),
                m = math.floor((s.worth or s.metal or 1) * 1000 + 0.5) / 1000,
            }
        end
        log("metal spots from WG resource_spot_finder: " .. #spots)
        return spots
    end

    local mw, mh = Spring.GetMetalMapSize()
    if not mw or mw == 0 then return {} end

    local scaleX = Game.mapSizeX / mw
    local scaleZ = Game.mapSizeZ / mh

    local max_v = 0
    for mx = 0, mw - 1 do
        for mz = 0, mh - 1 do
            local v = Spring.GetMetalAmount(mx, mz) or 0
            if v > max_v then max_v = v end
        end
    end
    if max_v <= 0 then return {} end
    local threshold = max_v * 0.5

    local spots, visited = {}, {}
    for mx = 0, mw - 1 do
        for mz = 0, mh - 1 do
            local key = mx * (mh + 1) + mz
            if not visited[key] then
                local v = Spring.GetMetalAmount(mx, mz) or 0
                if v >= threshold then
                    local is_peak = true
                    for dx = -1, 1 do
                        for dz = -1, 1 do
                            local nx, nz = mx + dx, mz + dz
                            if nx >= 0 and nx < mw and nz >= 0 and nz < mh then
                                if (Spring.GetMetalAmount(nx, nz) or 0) > v then
                                    is_peak = false
                                end
                            end
                        end
                    end
                    if is_peak then
                        spots[#spots + 1] = {
                            x = math.floor((mx + 0.5) * scaleX),
                            z = math.floor((mz + 0.5) * scaleZ),
                            m = math.floor(v * 1000 + 0.5) / 1000,
                        }
                        for dx = -3, 3 do
                            for dz = -3, 3 do
                                visited[(mx + dx) * (mh + 1) + (mz + dz)] = true
                            end
                        end
                    end
                end
            end
        end
    end
    log("metal spots from metal map scan: " .. #spots)
    return spots
end

local function connect()
    if socket == nil then
        log("FATAL: LuaSocket not available. Set LuaSocketEnabled=1 in springsettings.cfg "
            .. "or check that this engine build includes LuaSocket.")
        return false
    end
    sock = socket.tcp()
    sock:settimeout(10)  -- seconds; blocks engine but bounds hangs
    local ok, err = sock:connect(HOST, PORT)
    if not ok then
        log("connect to " .. HOST .. ":" .. PORT .. " failed: " .. tostring(err))
        sock:close(); sock = nil
        return false
    end
    log("connected to Python @ " .. HOST .. ":" .. PORT)

    -- Handshake
    local sent, serr = send_msg(sock, { type = "hello", frame = Spring.GetGameFrame() })
    if not sent then log("hello send failed: " .. tostring(serr)); return false end
    local hello, herr = recv_msg(sock)
    if not hello then log("hello recv failed: " .. tostring(herr)); return false end
    log("handshake OK (server said: " .. tostring(hello.type) .. ")")

    -- Crank the simulation speed. Engine default is 1x, but headless can
    -- run far faster. For RL training / replay extraction this is the
    -- difference between 30 minutes wall-clock and 90 seconds. The actual
    -- ceiling is the `maxspeed` modoption baked into the startscript:
    -- our generated startscripts set 100x; replay startscripts inherit
    -- whatever the original game used (typically 10-20x for online play).
    if GAME_SPEED and GAME_SPEED > 1 then
        log("setting gamespeed=" .. tostring(GAME_SPEED))
        Spring.SendCommands("setminspeed " .. tostring(GAME_SPEED))
        Spring.SendCommands("setmaxspeed " .. tostring(GAME_SPEED))
        Spring.SendCommands("speed " .. tostring(GAME_SPEED))
    end

    connected = true
    return true
end

local function do_step(frame, done, result)
    local obs = build_observation(frame, done, result)
    local ok, err = send_msg(sock, obs)
    if not ok then log("send obs failed: " .. tostring(err)); quitting = true; return end

    if done then return end  -- terminal obs: no actions expected

    local msg, rerr = recv_msg(sock)
    if not msg then log("recv action failed: " .. tostring(rerr)); quitting = true; return end

    if msg.type == "actions" then
        local n = apply_actions(msg.actions)
        steps_done = steps_done + 1
        log("step " .. steps_done .. " frame=" .. frame .. " applied=" .. n)
    elseif msg.type == "quit" then
        log("server requested quit at step " .. steps_done)
        quitting = true
    else
        log("unknown msg type: " .. tostring(msg.type))
    end
end

-- Replay-mode tick: send god's-eye obs + captured commands, no recv.
-- Python is purely a sink during replay extraction.
local function do_replay_step(frame, done, result)
    -- Detach the current buffer so any UnitCommand callins firing during the
    -- send don't pollute the next tick's batch.
    local commands = command_buffer
    command_buffer = {}
    local obs = build_observation_replay(frame, done, result, commands)
    local ok, err = send_msg(sock, obs)
    if not ok then log("send replay obs failed: " .. tostring(err)); quitting = true; return end
    steps_done = steps_done + 1
    if steps_done % 30 == 0 or done then
        log(("replay step %d frame=%d cmds=%d"):format(steps_done, frame, #commands))
    end
end

--------------------------------------------------------------------------------
-- Spring callins
--------------------------------------------------------------------------------

function widget:Initialize()
    load_config()
    MY_TEAM_ID = Spring.GetMyTeamID()
    MY_ALLY_ID = Spring.GetMyAllyTeamID()
    -- METAL_SPOTS is computed lazily on the first GameFrame so that
    -- WG['resource_spot_finder'] has had time to populate from its own Initialize.
    log(("init: mode=%s host=%s port=%d step_interval=%d team=%s ally=%s"):format(
        MODE, HOST, PORT, STEP_INTERVAL, tostring(MY_TEAM_ID), tostring(MY_ALLY_ID)))
    -- Defer the actual connect until the first Update — gives Python a moment
    -- to be ready after spawning the headless subprocess.
end

-- Replay-mode capture hook. Fires whenever a unit receives a command -- in a
-- replay these come from the demo stream, in live games they come from
-- whoever is playing that team. We append to the per-frame buffer.
function widget:UnitCommand(unitID, unitDefID, unitTeam, cmdID, cmdParams, cmdOptions, cmdTag)
    if MODE ~= "replay" then return end
    -- cmdParams is a Lua array of numbers; serialize as a plain list.
    local params = {}
    if type(cmdParams) == "table" then
        for i, v in ipairs(cmdParams) do params[i] = v end
    end
    -- cmdOptions can be a table or a number bitmask depending on engine version.
    local opts = {}
    if type(cmdOptions) == "table" then
        for k, v in pairs(cmdOptions) do
            if type(k) == "string" then opts[k] = v end
        end
    elseif type(cmdOptions) == "number" then
        opts.bits = cmdOptions
    end
    command_buffer[#command_buffer + 1] = {
        frame = Spring.GetGameFrame(),
        unit = unitID,
        def = unitDefID,
        team = unitTeam,
        cmd_id = cmdID,
        params = params,
        options = opts,
        tag = cmdTag,
    }
end

-- GameOver fires when the engine declares the match finished.
-- In live mode: send a terminal observation (win or loss) so Python gets a
--   clean done=True rather than a ProtocolError when the connection drops.
-- In replay mode: flush the final observation and mark for quit.
function widget:GameOver(winningAllyTeams)
    if MODE == "replay" then
        log("GameOver received during replay -- finishing extraction")
        if connected and sock then
            local f = Spring.GetGameFrame()
            pcall(function()
                send_msg(sock, build_observation_replay(f, true, "game_over", command_buffer))
            end)
            command_buffer = {}
        end
        quitting = true
    else
        -- Live mode. sock may already be nil if start_quit() fired first.
        -- Note: widgetHandler passes no args to w:GameOver(), so winningAllyTeams
        -- is always nil. Infer win/loss: if our team still has a commander-range
        -- unit (max_hp 2000-4500), our commander survived → we won.
        -- Works for both deathmode=com and deathmode=killall.
        if connected and sock then
            local f       = Spring.GetGameFrame()
            local my_team = MY_TEAM_ID or Spring.GetMyTeamID()
            local has_com = false
            for _, uid in ipairs(Spring.GetTeamUnits(my_team) or {}) do
                local _, max_hp = Spring.GetUnitHealth(uid)
                if max_hp and max_hp > 2000 and max_hp < 4500 then
                    has_com = true; break
                end
            end
            local result = has_com and "win" or "loss"
            log("GameOver: result=" .. result)
            pcall(function()
                send_msg(sock, build_observation(f, true, result))
            end)
        end
        quitting = true
    end
end

-- Two-phase quit. Phase 1 declares game-over and waits ~30 frames for the
-- game_end.lua gadget to run and the engine's demo writer to flush; phase 2
-- issues `quitforce` (note: NOT `quit`, which BAR doesn't honor in many
-- contexts) to actually exit the process.
local function start_quit()
    if quit_phase ~= nil then return end
    if sock then pcall(function() sock:close() end); sock = nil end
    log("starting quit sequence (GameOver -> quitforce)")
    local my_ally = Spring.GetMyAllyTeamID()
    -- Declare ourselves the winner. This fires gameover events and lets the
    -- demo writer finalize its file before we kill the process.
    pcall(function() Spring.GameOver({ my_ally }) end)
    quit_phase = "go"
    quit_phase_until_frame = Spring.GetGameFrame() + 60  -- 2 s for game_end.lua + demo writer
end

local function tick_quit(frame)
    if quit_phase == "go" and frame >= quit_phase_until_frame then
        log("forcing engine exit (quitforce)")
        Spring.SendCommands("quitforce")
        quit_phase = "done"
    end
end

function widget:Update(dt)
    if quitting and quit_phase == nil then
        start_quit()
        return
    end
    if not connected and quit_phase == nil then
        if not connect() then
            -- Could not connect; try again on the next Update tick.
            return
        end
    end
end

function widget:GameFrame(frame)
    -- Lazy-init metal spots on the first frame so WG widgets have had time
    -- to run their own Initialize() and populate WG['resource_spot_finder'].
    if METAL_SPOTS == nil then
        METAL_SPOTS = compute_metal_spots()
    end
    if quit_phase ~= nil then
        tick_quit(frame)
        return
    end
    if not connected or quitting then return end
    if frame - last_step_frame < STEP_INTERVAL then return end
    last_step_frame = frame
    if MODE == "replay" then
        do_replay_step(frame, false, nil)
        -- max_steps is a safety cap in replay mode too: if a replay is
        -- absurdly long or GameOver never fires, we don't hang forever.
        if MAX_STEPS > 0 and steps_done >= MAX_STEPS then
            do_replay_step(frame, true, "max_steps")
            quitting = true
        end
        return
    end
    do_step(frame, false, nil)
    if steps_done >= MAX_STEPS then
        -- Send a terminal obs as a courtesy and then quit.
        do_step(frame, true, "max_steps")
        quitting = true
    end
end

function widget:Shutdown()
    if sock then pcall(function() sock:close() end); sock = nil end
end
