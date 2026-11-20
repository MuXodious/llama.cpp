#include "parsers.h"

// Instella-MoE (AMD, DeepSeek-V3 derivative) format:
// - pure reasoning model: an assistant turn is
//   <｜Assistant｜>[\n]* + open tag ... + close tag + ...content
//   the think-tag is a plain-text XML-style pair (no special tokens) and the chat
//   template never mentions it, so the autoparser can never detect reasoning on its own
// - the turn is closed by <｜end▁of▁sentence｜> (eos); tool calls use the
//   DeepSeek-R1-Distill-Qwen <｜tool▁calls▁begin｜> syntax (the model rarely uses them)
common_chat_params common_chat_params_init_instella(const common_chat_template &          tmpl,
                                                    const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    const std::string GEN_PREFIX  = "<｜Assistant｜>";
    const std::string THINK_START = "<" "think" ">";
    const std::string THINK_END   = "</" "think" ">";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tags  = { THINK_END };

    data.preserved_tokens = { THINK_START, THINK_END };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, "<｜Assistant｜>" },
        { COMMON_CHAT_ROLE_USER,      "<｜User｜>"      },
    };

    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = GEN_PREFIX + THINK_START + "\n" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += THINK_END + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PREFIX);

        auto think_body  = p.until_one_of(data.thinking_end_tags);
        auto think_block = p.optional(p.space() + THINK_START + p.space() +
                                      p.ac(p.reasoning(think_body) + THINK_END, data.thinking_end_tags));
        auto reasoning   = extract_reasoning ? think_block : p.eps();

        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    return data;
}
