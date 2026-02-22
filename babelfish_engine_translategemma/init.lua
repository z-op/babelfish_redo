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
    gpu_enabled = nil,
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
    -- Simplified for brevity in logs
    return "{ " .. table.concat(summary, ", ") .. " }"
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
            if data then core.log("action", "[babelfish_engine_translategemma] Ollama version: " .. (data.version or "unknown")) end
        end
    end)

    -- Model Info
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
    -- Простой промпт, так как язык уже известен
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
-- Language Detection (New function)
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

    -- Build request based on API type
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
        -- Fallback or HF
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

                    -- Anti-hallucination check
                    if code == "en" then
                         if query:find(string.char(208)) or query:find(string.char(209)) then
                            core.log("warning", "[babelfish_engine_translategemma] Detection conflict: model says 'en' but Cyrillic found. Overriding to 'ru'.")
                            code = "ru"
                         end
                    end
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

        local translation = parse_response(api_type, data)

        if translation then
            translation = translation:match("^%s*(.-)%s*$")
            return callback(true, translation, source)
        end

        return callback(false, S("Error parsing translation response"))
    end)
end

-- ============================================================================
-- Public API Function
-- ============================================================================

local function translate(source, target, query, callback)
    if engine_status == "error" then return callback(false, S("Engine error while initializing."))
    elseif engine_status == "init" then return callback(false, S("Engine not yet initialized.")) end

    -- Handle Chinese variants
    if source == "zh_HANT" or source == "zh-tw" or source == "zh-hant" then source = "zh" end
    if target == "zh_HANT" or target == "zh-tw" or target == "zh-hant" then target = "zh" end

    if source == "auto" then
        return detect_language(query, function(detected)
            if detected == target then
                -- Optimization: Source is same as target, skip translation
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
