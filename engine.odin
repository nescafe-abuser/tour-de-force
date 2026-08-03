package tdf

import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

read_stdin :: proc(guide: string, buffer: []u8) -> string {
	fmt.print(guide)

	bytes_read, err := os.read(os.stdin, buffer)
	if err != nil || bytes_read <= 0 {
		return ""
	}

	line := string(buffer[:bytes_read])
	line = strings.trim_right(line, "\r\n")
	return line
}
generate :: proc(t: ^Transformer, tok: ^Tokenizer, s: ^Sampler, prompt: string, steps: int) {
	prompt_str := prompt if len(prompt) > 0 else ""

	prompt_tokens := encode(tok, prompt_str, true, false)
	defer delete(prompt_tokens)

	if len(prompt_tokens) < 1 {
		fmt.eprintln("Error: Expected at least 1 prompt token.")
		return
	}

	start_time: time.Tick
	next_token: i32
	token := prompt_tokens[0]
	pos := 0

	max_steps := steps
	if max_steps == 0 || max_steps > int(t.config.seq_len) {
		max_steps = int(t.config.seq_len)
	}

	for pos < max_steps {
		logits := forward(t, token, i32(pos))

		if pos < len(prompt_tokens) - 1 {
			next_token = prompt_tokens[pos + 1]
		} else {
			next_token = sample(s, logits)
		}

		pos += 1

		if next_token == 1 || next_token == 2 {
			break
		}

		piece := decode(tok, token, next_token)
		safe_print_piece(piece)

		token = next_token

		if pos == 1 {
			start_time = time.tick_now()
		}
	}

	fmt.println()

	if pos > 1 {
		elapsed := time.tick_diff(start_time, time.tick_now())
		seconds := time.duration_seconds(elapsed)
		if seconds > 0 {
			tok_per_sec := f64(pos - 1) / seconds
			fmt.eprintfln("\nAchieved throughput: %.2f tokens/sec", tok_per_sec)
		}
	}
}

chat :: proc(
	t: ^Transformer,
	tok: ^Tokenizer,
	s: ^Sampler,
	cli_user_prompt: string,
	cli_system_prompt: string,
	steps: int,
) {
	stdin_buffer: [1024]u8
	system_prompt := cli_system_prompt
	user_prompt := cli_user_prompt

	user_turn := true
	next_token: i32
	token: i32
	pos := 0

	prompt_tokens: []i32
	user_idx := 0

	max_steps := steps
	if max_steps == 0 || max_steps > int(t.config.seq_len) {
		max_steps = int(t.config.seq_len)
	}

	for pos < max_steps {
		if user_turn {
			if pos == 0 && len(system_prompt) == 0 {
				system_prompt = read_stdin("Enter system prompt (optional): ", stdin_buffer[:])
			}

			if pos == 0 && len(user_prompt) > 0 {
			} else {
				user_prompt = read_stdin("User: ", stdin_buffer[:])
			}

			rendered_prompt: string
			if pos == 0 && len(system_prompt) > 0 {
				rendered_prompt = fmt.tprintf(
					"[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
					system_prompt,
					user_prompt,
				)
			} else {
				rendered_prompt = fmt.tprintf("[INST] %s [/INST]", user_prompt)
			}

			if len(prompt_tokens) > 0 {
				delete(prompt_tokens)
			}

			prompt_tokens = encode(tok, rendered_prompt, true, false)
			user_idx = 0
			user_turn = false
			fmt.print("Assistant: ")
		}

		if user_idx < len(prompt_tokens) {
			token = prompt_tokens[user_idx]
			user_idx += 1
		} else {
			token = next_token
		}

		if token == 2 {
			user_turn = true
			fmt.println()
			continue
		}

		logits := forward(t, token, i32(pos))
		next_token = sample(s, logits)
		pos += 1

		if user_idx >= len(prompt_tokens) && next_token != 2 {
			piece := decode(tok, token, next_token)
			safe_print_piece(piece)
		}
	}

	if len(prompt_tokens) > 0 {
		delete(prompt_tokens)
	}

	fmt.println()
}

