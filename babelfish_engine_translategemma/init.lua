-- babelfish_redo/babelfish_engine_translategemma/init.lua
-- Translation via TranslateGemma (Gemma 3 based open translation model)
-- Based on babelfish_engine_lingva by Tai "DuCake" Kedzierski and 1F616EMO
-- Adapted for TranslateGemma with z.ai by Evgeniy aka Bad
-- SPDX-License-Identifier: LGPL-3.0-or-later

local http = assert(core.request_http_api(),
    "Could not get HTTP API table. Add babelfish_engine_translategemma to secure.http_mods")

local S = core.get_translator("babelfish_engine_translategemma")

local engine_status = "init"
local language_codes = {}
local language_alias = {}

-- Store Ollama info for logging
local ollama_info = {
    version = nil,
    model_info = nil,
    gpu_enabled = "Unknown",
    vram_checked = false, -- Flag to check VRAM only once per model load
}

-- Configuration
local api_type = core.settings:get("babelfish_engine_translategemma.api_type") or "ollama"
local serviceurl = core.settings:get("babelfish_engine_translategemma.serviceurl")
local model_name = core.settings:get("babelfish_engine_translategemma.model") or "translategemma"

-- Debug logging settings
local debug_logging = core.settings:get_bool("babelfish_engine_translategemma.debug", false)
local verbose_debug = core.settings:get_bool("babelfish_engine_translategemma.verbose_debug", false)
local show_license = core.settings:get_bool("babelfish_engine_translategemma.show_license", false)

-- Default URLs based on API type
if not serviceurl then
    if api_type == "ollama" then
        serviceurl = "http://localhost:11434/api/generate"
    elseif api_type == "vllm" then
        serviceurl = "http://localhost:8000/v1/completions"
    elseif api_type == "hf" then
        serviceurl = "https://api-inference.huggingface.co/models/google/translategemma-4b"
    elseif api_type == "openai" then
        serviceurl = "http://localhost:8000/v1/chat/completions"
    else
        serviceurl = "http://localhost:11434/api/generate"
    end
    core.log("warning",
        "[babelfish_engine_translategemma] babelfish_engine_translategemma.serviceurl not specified, " ..
        "using default for " .. api_type .. ": " .. serviceurl)
end

-- ============================================================================
-- Ollama Info Functions (Log version, GPU, etc.)
-- ============================================================================

local function get_ollama_base_url()
    if api_type ~= "ollama" then return nil end
    local base = serviceurl:match("^(https?://[^/]+)")
    return base or "http://localhost:11434"
end

local function format_bytes(bytes)
    if not bytes then return "N/A" end
    if bytes >= 1e12 then return string.format("%.2f TB", bytes / 1e12)
    elseif bytes >= 1e9 then return string.format("%.2f GB", bytes / 1e9)
    elseif bytes >= 1e6 then return string.format("%.2f MB", bytes / 1e6)
    elseif bytes >= 1e3 then return string.format("%.2f KB", bytes / 1e3)
    else return string.format("%d B", bytes) end
end

local function get_model_info_summary(model_info)
    if not model_info then return "N/A" end
    local summary = {}
    if model_info.general then
        local general = {}
        if model_info.general.architecture then table.insert(general, "architecture=" .. tostring(model_info.general.architecture)) end
        if model_info.general.file_type then table.insert(general, "file_type=" .. tostring(model_info.general.file_type)) end
        if model_info.general.parameter_count then
            local params = model_info.general.parameter_count
            table.insert(general, string.format("parameters=%.1fB", params / 1e9))
        end
        if #general > 0 then table.insert(summary, "general: {" .. table.concat(general, ", ") .. "}") end
    end
    return "{ " .. table.concat(summary, ", ") .. " }"
end

-- Function to check running models and their memory usage (VRAM vs RAM)
local function check_ollama_vram_usage()
    if api_type ~= "ollama" then return end

    http.fetch({
        url = get_ollama_base_url() .. "/api/ps",
        method = "GET",
        timeout = 2
    }, function(res)
        if res.succeeded and res.data then
            local data = core.parse_json(res.data)
            if data and data.models then
                -- Iterate over running models to find ours
                for _, m in ipairs(data.models) do
                    -- Match by name (Ollama returns "name" field in /api/ps)
                    if m.name == model_name or m.model == model_name or m.name:find("^" .. model_name) then
                        local vram = m.size_vram or 0
                        local total = m.size or 0

                        local vram_str = format_bytes(vram)
                        local total_str = format_bytes(total)

                        core.log("action", "[babelfish_engine_translategemma] ========== Model Memory Usage ==========")
                        core.log("action", "[babelfish_engine_translategemma] Model loaded: " .. m.name)
                        core.log("action", string.format("[babelfish_engine_translategemma] Total Memory: %s | VRAM: %s", total_str, vram_str))

                        if total > 0 then
                            local percent = (vram / total) * 100
                            if vram == 0 then
                                core.log("action", "[babelfish_engine_translategemma] Mode: CPU Only (0% offloaded to GPU)")
                                ollama_info.gpu_enabled = "CPU"
                            elseif percent >= 99 then
                                core.log("action", "[babelfish_engine_translategemma] Mode: Full GPU (100% offloaded)")
                                ollama_info.gpu_enabled = "Full GPU"
                            else
                                core.log("action", string.format("[babelfish_engine_translategemma] Mode: Hybrid (%.1f%% offloaded to GPU)", percent))
                                ollama_info.gpu_enabled = "Hybrid GPU/CPU"
                            end
                            -- Detailed layer info isn't strictly available via API, but size_vram implies layers offloaded.
                            core.log("action", "[babelfish_engine_translategemma] Layers info: Offloaded memory indicates layer distribution in VRAM.")
                        end
                        core.log("action", "[babelfish_engine_translategemma] =====================================")
                        ollama_info.vram_checked = true
                        return -- Stop after finding our model
                    end
                end
                -- If model not found in list (shouldn't happen immediately after request, but possible due to timing)
                if not ollama_info.vram_checked then
                     core.log("action", "[babelfish_engine_translategemma] Model not found in active list (might be unloading or timing issue).")
                end
            end
        end
    end)
end

local function log_ollama_startup_info()
    if api_type ~= "ollama" then return end
    core.log("action", "[babelfish_engine_translategemma] ========== Ollama Startup Info ==========")
    core.log("action", "[babelfish_engine_translategemma] Ollama API URL: " .. get_ollama_base_url())
    core.log("action", "[babelfish_engine_translategemma] Model name: " .. model_name)

    -- Version
    http.fetch({ url = get_ollama_base_url() .. "/api/version", method = "GET", timeout = 5 }, function(res)
        if res.succeeded then
            local data = core.parse_json(res.data)
            if data then
                ollama_info.version = data.version
                core.log("action", "[babelfish_engine_translategemma] Ollama version: " .. (data.version or "unknown"))
            end
        end
    end)

    -- Model Info (Static details)
    http.fetch({
        url = get_ollama_base_url() .. "/api/show",
        method = "POST",
        extra_headers = { "Content-Type: application/json" },
        post_data = core.write_json({ name = model_name }),
        timeout = 10
    }, function(res)
        if res.succeeded then
            local data = core.parse_json(res.data)
            if data and data.details then
                 core.log("action", "[babelfish_engine_translategemma] Model format: " .. (data.details.format or "N/A"))
                 core.log("action", "[babelfish_engine_translategemma] Quantization: " .. (data.details.quantization_level or "N/A"))
                 if data.details.family then
                    core.log("action", "[babelfish_engine_translategemma] Model family: " .. data.details.family)
                 end
            end
        end
    end)

    core.log("action", "[babelfish_engine_translategemma] =======================================")
end

-- ============================================================================
-- Language Definitions
-- ============================================================================

local supported_languages = {
    "ar", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es",
    "et", "fa", "fi", "fr", "he", "hi", "hr", "hu", "id", "it",
    "ja", "ko", "lt", "lv", "ms", "nl", "no", "pl", "pt", "ro",
    "ru", "sk", "sl", "sr", "sv", "th", "tl", "tr", "uk", "vi",
    "zh", "af", "am", "az", "be", "bs", "cy", "eo", "eu", "gl",
    "gu", "hy", "ka", "kn", "ml", "mr", "ne", "pa", "si", "sq",
    "sw", "ta", "te", "ur", "uz"
}

local language_names = {
    ["ar"] = "Arabic", ["bg"] = "Bulgarian", ["bn"] = "Bengali", ["ca"] = "Catalan",
    ["cs"] = "Czech", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek",
    ["en"] = "English", ["es"] = "Spanish", ["et"] = "Estonian", ["fa"] = "Persian",
    ["fi"] = "Finnish", ["fr"] = "French", ["he"] = "Hebrew", ["hi"] = "Hindi",
    ["hr"] = "Croatian", ["hu"] = "Hungarian", ["id"] = "Indonesian", ["it"] = "Italian",
    ["ja"] = "Japanese", ["ko"] = "Korean", ["lt"] = "Lithuanian", ["lv"] = "Latvian",
    ["ms"] = "Malay", ["nl"] = "Dutch", ["no"] = "Norwegian", ["pl"] = "Polish",
    ["pt"] = "Portuguese", ["ro"] = "Romanian", ["ru"] = "Russian", ["sk"] = "Slovak",
    ["sl"] = "Slovenian", ["sr"] = "Serbian", ["sv"] = "Swedish", ["th"] = "Thai",
    ["tl"] = "Tagalog", ["tr"] = "Turkish", ["uk"] = "Ukrainian", ["vi"] = "Vietnamese",
    ["zh"] = "Chinese", ["af"] = "Afrikaans", ["am"] = "Amharic", ["az"] = "Azerbaijani",
    ["be"] = "Belarusian", ["bs"] = "Bosnian", ["cy"] = "Welsh", ["eo"] = "Esperanto",
    ["eu"] = "Basque", ["gl"] = "Galician", ["gu"] = "Gujarati", ["hy"] = "Armenian",
    ["ka"] = "Georgian", ["kn"] = "Kannada", ["ml"] = "Malayalam", ["mr"] = "Marathi",
    ["ne"] = "Nepali", ["pa"] = "Punjabi", ["si"] = "Sinhala", ["sq"] = "Albanian",
    ["sw"] = "Swahili", ["ta"] = "Tamil", ["te"] = "Telugu", ["ur"] = "Urdu",
    ["uz"] = "Uzbek"
}

do
    local valid_alias = {
        ["zh"] = { "zhs", "zh-cn", "zh-hans", "chinese" },
        ["en"] = { "english" },
        ["ru"] = { "russian" },
        ["de"] = { "german" },
        ["fr"] = { "french" },
        ["es"] = { "spanish" },
        ["ja"] = { "japanese" },
        ["ko"] = { "korean" },
    }
    for _, code in ipairs(supported_languages) do
        language_codes[code] = language_names[code] or code
        if valid_alias[code] then
            for _, alias in ipairs(valid_alias[code]) do language_alias[alias] = code end
        end
    end
    engine_status = "ready"
    log_ollama_startup_info()
end

-- ============================================================================
-- Prompt Building
-- ============================================================================

local function build_translation_prompt(source, target, query)
    local source_lang = language_names[source] or source
    local target_lang = language_names[target] or target
    return string.format(
        "Translate the following text from %s to %s. Provide only the translation, no explanations:\n\n%s",
        source_lang, target_lang, query
    )
end

-- ============================================================================
-- Response Parsing (Support for multiple APIs)
-- ============================================================================

local function parse_response(api_type, data)
    if api_type == "ollama" then
        return data.response
    elseif api_type == "vllm" or api_type == "openai" then
        if data.choices and data.choices[1] then
            return data.choices[1].text or data.choices[1].message.content
        end
    elseif api_type == "hf" then
        if type(data) == "table" and data[1] and data[1].generated_text then return data[1].generated_text end
    end
    return nil
end

-- ============================================================================
-- Language Detection
-- ============================================================================

local function detect_language(query, callback)
    local prompt = string.format(
        "Identify the language of the following text. " ..
        "Reply ONLY with the ISO 639-1 language code (e.g., 'en', 'ru', 'de'). " ..
        "Do not add punctuation or explanations.\n\nText: %s",
        query
    )

    local post_data
    local extra_headers = { "Content-Type: application/json" }

    if api_type == "ollama" then
        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            stream = false,
            options = { num_predict = 5, temperature = 0.1 }
        })
    elseif api_type == "openai" then
        post_data = core.write_json({
            model = model_name,
            messages = { { role = "user", content = prompt } },
            max_tokens = 5,
            temperature = 0.1
        })
    elseif api_type == "vllm" then
        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            max_tokens = 5,
            temperature = 0.1
        })
    else
        post_data = core.write_json({ inputs = prompt, parameters = { max_new_tokens = 5 } })
    end

    http.fetch({
        url = serviceurl,
        method = "POST",
        timeout = 10,
        extra_headers = extra_headers,
        post_data = post_data,
    }, function(response)
        if not response.succeeded then
            core.log("warning", "[babelfish_engine_translategemma] Language detection request failed.")
            return callback("auto")
        end

        local data = core.parse_json(response.data)
        if data then
            local raw_response = parse_response(api_type, data)
            if raw_response then
                local code = raw_response:match("^%s*(%a%a%a?)%s*$")
                if code then
                    code = code:lower()
                    core.log("action", "[babelfish_engine_translategemma] Detected language: " .. code)
                    return callback(code)
                end
            end
        end
        return callback("auto")
    end)
end

-- ============================================================================
-- Main Translation Request
-- ============================================================================

local function make_request(source, target, query, callback)
    local prompt = build_translation_prompt(source, target, query)
    local post_data
    local extra_headers = { "Content-Type: application/json" }

    if api_type == "ollama" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local num_predict = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024
        local num_ctx = tonumber(core.settings:get("babelfish_engine_translategemma.num_ctx")) or 2048

        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            stream = false,
            options = {
                temperature = temperature,
                num_predict = num_predict,
                num_ctx = num_ctx,
            }
        })
    elseif api_type == "vllm" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local max_tokens = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024
        post_data = core.write_json({
            model = model_name,
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = temperature,
        })
    elseif api_type == "openai" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local max_tokens = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024
        post_data = core.write_json({
            model = model_name,
            messages = { { role = "user", content = prompt } },
            max_tokens = max_tokens,
            temperature = temperature,
        })
    elseif api_type == "hf" then
        local temperature = tonumber(core.settings:get("babelfish_engine_translategemma.temperature")) or 0.3
        local max_tokens = tonumber(core.settings:get("babelfish_engine_translategemma.num_predict")) or 1024
        post_data = core.write_json({
            inputs = prompt,
            parameters = { max_new_tokens = max_tokens, temperature = temperature, return_full_text = false }
        })
        local hf_token = core.settings:get("babelfish_engine_translategemma.hf_token")
        if hf_token then table.insert(extra_headers, "Authorization: Bearer " .. hf_token) end
    else
        return callback(false, S("Unsupported API type"))
    end

    http.fetch({
        url = serviceurl,
        method = "POST",
        timeout = 30,
        extra_headers = extra_headers,
        post_data = post_data,
    }, function(response)
        if not response.succeeded then
            return callback(false, S("HTTP request failed"))
        end

        local data = core.parse_json(response.data)
        if not data then
            return callback(false, S("JSON parse error"))
        end

        -- Log Speed Stats (Optional, keeps previous functionality)
        if api_type == "ollama" and data.eval_count and data.eval_duration then
            local eval_duration_sec = data.eval_duration / 1e9
            if eval_duration_sec > 0 then
                local tokens_per_sec = data.eval_count / eval_duration_sec
                -- Only log speed, mode is determined by VRAM check now
                core.log("action", string.format("[babelfish_engine_translategemma] Inference speed: %.2f tokens/sec", tokens_per_sec))
            end
        end

        -- Check VRAM usage after first successful request
        if api_type == "ollama" and not ollama_info.vram_checked then
            check_ollama_vram_usage()
        end

        local translation = parse_response(api_type, data)

        if translation then
            translation = translation:match("^%s*(.-)%s*$")
            return callback(true, translation, source)
        end

        return callback(false, S("Error parsing translation response"))
    end)
end

-- ============================================================================
-- Client Language Helper
-- ============================================================================

-- Helper to guess language based on script to avoid AI detection if possible
local function simple_script_check(text, lang_code)
    -- Cyrillic check for Russian and other Cyrillic languages
    if lang_code == "ru" or lang_code == "uk" or lang_code == "bg" or lang_code == "sr" then
        if text:find(string.char(208)) or text:find(string.char(209)) then
            return true
        end
    end

    -- Basic Latin check (English, German, French, etc.)
    -- Be careful: many languages use Latin script. We only return true if we are sure
    -- it matches the expected alphabet in a trivial case.
    if lang_code == "en" or lang_code == "de" or lang_code == "fr" then
        -- Check if it's strictly ASCII (basic latin) for English mostly
        -- This is a weak check, but helps with simple English messages
        if not text:find("[\128-\255]") then
            return true
        end
    end

    return false
end

local function get_client_language(player_name)
    if not player_name then return nil end

    -- Use core.get_player_information if available (MT 5.4+)
    -- Note: This might not be available in all contexts or might need async handling in MT 5.x
    -- But usually inside a chat message hook, the player is online.

    local info = core.get_player_information(player_name)
    if info and info.lang_code and info.lang_code ~= "" then
        -- Normalize (e.g., en_US -> en)
        local main_lang = info.lang_code:sub(1, 2):lower()
        if language_codes[main_lang] then
            return main_lang
        end
    end

    return nil
end

-- ============================================================================
-- Public API Function
-- ============================================================================

-- Added optional player_name argument
local function translate(source, target, query, callback, player_name)
    if engine_status == "error" then return callback(false, S("Engine error while initializing."))
    elseif engine_status == "init" then return callback(false, S("Engine not yet initialized.")) end

    -- Handle Chinese variants
    if source == "zh_HANT" or source == "zh-tw" or source == "zh-hant" then source = "zh" end
    if target == "zh_HANT" or target == "zh-tw" or target == "zh-hant" then target = "zh" end

    if source == "auto" then
        -- Try to optimize detection using client language
        local client_lang = get_client_language(player_name)

        if client_lang and client_lang ~= target then
            -- Heuristic: check if text script matches the client language
            if simple_script_check(query, client_lang) then
                core.log("action", string.format("[babelfish_engine_translategemma] Using client language '%s' for player '%s' (script match).", client_lang, player_name or "unknown"))
                return make_request(client_lang, target, query, callback)
            else
                -- Script mismatch or unclear, fallback to AI detection but hint the client language
                -- We log that we are falling back to detection
                core.log("action", string.format("[babelfish_engine_translategemma] Client lang is '%s' but text script differs. Using AI detection.", client_lang))
            end
        end

        -- Fallback to AI detection
        return detect_language(query, function(detected)
            if detected == target then
                return callback(true, query, detected)
            else
                return make_request(detected, target, query, callback)
            end
        end)
    else
        return make_request(source, target, query, callback)
    end
end

-- ============================================================================
-- Mapping & Registration
-- ============================================================================

local mt_language_map = {
    ["es_US"] = "es", ["lzh"] = "zh", ["zh_CN"] = "zh", ["zh_TW"] = "zh",
    ["sr_Cyrl"] = "sr", ["sr_Latn"] = "sr", ["ru_RU"] = "ru",
    ["en_US"] = "en", ["en_GB"] = "en",
    ["de_DE"] = "de", ["de_AT"] = "de", ["de_CH"] = "de",
    ["fr_FR"] = "fr", ["fr_CA"] = "fr", ["fr_CH"] = "fr",
    ["it_IT"] = "it", ["it_CH"] = "it",
    ["es_ES"] = "es", ["es_MX"] = "es", ["es_AR"] = "es",
    ["pt_PT"] = "pt", ["pt_BR"] = "pt",
    ["nl_NL"] = "nl", ["nl_BE"] = "nl",
    ["pl_PL"] = "pl", ["uk_UA"] = "uk", ["cs_CZ"] = "cs", ["sk_SK"] = "sk",
    ["hu_HU"] = "hu", ["ro_RO"] = "ro", ["bg_BG"] = "bg", ["hr_HR"] = "hr",
    ["sr_RS"] = "sr", ["sl_SI"] = "sl", ["et_EE"] = "et", ["lv_LV"] = "lv",
    ["lt_LT"] = "lt", ["el_GR"] = "el", ["fi_FI"] = "fi", ["sv_SE"] = "sv",
    ["da_DK"] = "da", ["no_NO"] = "no", ["nb_NO"] = "no", ["nn_NO"] = "no",
    ["tr_TR"] = "tr", ["sq_AL"] = "sq", ["bs_BA"] = "bs", ["ca_ES"] = "ca",
    ["eu_ES"] = "eu", ["gl_ES"] = "gl", ["mt_MT"] = "mt", ["cy_GB"] = "cy",
    ["ja_JP"] = "ja", ["ko_KR"] = "ko", ["zh_HK"] = "zh", ["zh_SG"] = "zh",
    ["ar_SA"] = "ar", ["ar_EG"] = "ar", ["he_IL"] = "he", ["hi_IN"] = "hi",
    ["bn_BD"] = "bn", ["bn_IN"] = "bn", ["fa_IR"] = "fa", ["ur_PK"] = "ur",
    ["ta_IN"] = "ta", ["te_IN"] = "te", ["mr_IN"] = "mr", ["gu_IN"] = "gu",
    ["kn_IN"] = "kn", ["ml_IN"] = "ml", ["pa_IN"] = "pa", ["pa_PK"] = "pa",
    ["th_TH"] = "th", ["vi_VN"] = "vi", ["id_ID"] = "id", ["ms_MY"] = "ms",
    ["tl_PH"] = "tl", ["sw_KE"] = "sw", ["ne_NP"] = "ne", ["si_LK"] = "si",
    ["af_ZA"] = "af", ["am_ET"] = "am", ["hy_AM"] = "hy", ["az_AZ"] = "az",
    ["be_BY"] = "be", ["ka_GE"] = "ka", ["kk_KZ"] = "kk", ["uz_UZ"] = "uz",
}

babelfish.register_engine({
    translate = translate,
    language_codes = language_codes,
    language_alias = language_alias,
    mt_language_map = mt_language_map,
    compliance = nil,
    engine_label = "TranslateGemma",
})

core.log("action", "[babelfish_engine_translategemma] Engine registered successfully with API type: " .. api_type)
