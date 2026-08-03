package tdf

import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"





Token_Index :: struct {
	str: string,
	id:  i32,
}

Tokenizer :: struct {
	vocab:            []string,
	vocab_scores:     []f32,
	sorted_vocab:     []Token_Index,
	vocab_size:       i32,
	max_token_length: u32,
	byte_pieces:      [512]u8, 
}





init_tokenizer :: proc(tokenizer_path: string, vocab_size: i32) -> (t: Tokenizer, ok: bool) {
	t.vocab_size = vocab_size
	t.vocab = make([]string, vocab_size)
	t.vocab_scores = make([]f32, vocab_size)

	
	for i in 0 ..< 256 {
		t.byte_pieces[i * 2] = u8(i)
		t.byte_pieces[i * 2 + 1] = 0
	}

	
	data, err := os.read_entire_file_from_path(tokenizer_path,context.allocator)
	if err != nil {
		fmt.eprintfln("Failed to load tokenizer file: %s", tokenizer_path)
		return t, false
	}
	defer delete(data)

	offset := 0

	
	t.max_token_length = (cast(^u32)&data[offset])^
	offset += size_of(u32)

	
	for i in 0 ..< int(vocab_size) {
		score := (cast(^f32)&data[offset])^
		offset += size_of(f32)
		t.vocab_scores[i] = score

		length := int((cast(^i32)&data[offset])^)
		offset += size_of(i32)

		str_bytes := data[offset:offset + length]
		offset += length

		
		t.vocab[i] = strings.clone_from_bytes(str_bytes)
	}

	
	t.sorted_vocab = make([]Token_Index, vocab_size)
	for i in 0 ..< int(vocab_size) {
		t.sorted_vocab[i] = Token_Index {
			str = t.vocab[i],
			id  = i32(i),
		}
	}

	
	slice.sort_by(t.sorted_vocab, proc(a, b: Token_Index) -> bool {
		return a.str < b.str
	})

	return t, true
}

free_tokenizer :: proc(t: ^Tokenizer) {
	for str in t.vocab {
		delete(str)
	}
	delete(t.vocab)
	delete(t.vocab_scores)
	delete(t.sorted_vocab)
}





str_lookup :: proc(str: string, sorted_vocab: []Token_Index) -> i32 {
	
	low := 0
	high := len(sorted_vocab) - 1

	for low <= high {
		mid := (low + high) / 2
		if sorted_vocab[mid].str == str {
			return sorted_vocab[mid].id
		} else if sorted_vocab[mid].str < str {
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	return -1
}

safe_print_piece :: proc(piece: string) {
	if len(piece) == 0 {
		return
	}

	
	if len(piece) == 1 {
		ch := piece[0]
		if !(ch >= 32 && ch <= 126) && ch != '\n' && ch != '\t' && ch != '\r' {
			return
		}
	}
	fmt.print(piece)
}


decode :: proc(t: ^Tokenizer, prev_token: i32, token: i32) -> string {
	piece := t.vocab[token]

	
	if prev_token == 1 && len(piece) > 0 && piece[0] == ' ' {
		piece = piece[1:]
	}

	
	if len(piece) == 6 && strings.has_prefix(piece, "<0x") && strings.has_suffix(piece, ">") {
		byte_val: u8
		hex_str := piece[3:5]

		
		val := 0
		for i in 0 ..< 2 {
			c := hex_str[i]
			val *= 16
			if c >= '0' &&
			   c <=
				   '9' {val += int(c - '0')} else if c >= 'A' && c <= 'F' {val += int(c - 'A' + 10)} else if c >= 'a' && c <= 'f' {val += int(c - 'a' + 10)}
		}
		byte_val = u8(val)

		
		return string(t.byte_pieces[byte_val * 2:byte_val * 2 + 1])
	}

	return piece
}





encode :: proc(t: ^Tokenizer, text: string, bos: bool, eos: bool) -> []i32 {
	tokens := make([dynamic]i32)

	if bos {
		append(&tokens, 1) 
	}

	
	if len(text) > 0 {
		dummy_prefix := str_lookup(" ", t.sorted_vocab)
		if dummy_prefix != -1 {
			append(&tokens, dummy_prefix)
		}
	}

	
	str_buffer: [256]u8
	str_len := 0

	for i := 0; i < len(text); i += 1 {
		c := text[i]

		
		if (c & 0xC0) != 0x80 {
			str_len = 0
		}

		str_buffer[str_len] = c
		str_len += 1

		
		if i + 1 < len(text) && (text[i + 1] & 0xC0) == 0x80 && str_len < 4 {
			continue
		}

		
		str_piece := string(str_buffer[0:str_len])
		id := str_lookup(str_piece, t.sorted_vocab)

		if id != -1 {
			append(&tokens, id)
		} else {
			
			for b_idx in 0 ..< str_len {
				append(&tokens, i32(str_buffer[b_idx]) + 3)
			}
		}
		str_len = 0
	}

	
	for {
		best_score: f32 = -1e10
		best_id: i32 = -1
		best_idx := -1

		
		for i in 0 ..< len(tokens) - 1 {
			tok1 := t.vocab[tokens[i]]
			tok2 := t.vocab[tokens[i + 1]]

			
			merged_str := strings.concatenate({tok1, tok2})
			defer delete(merged_str)

			id := str_lookup(merged_str, t.sorted_vocab)
			if id != -1 && t.vocab_scores[id] > best_score {
				best_score = t.vocab_scores[id]
				best_id = id
				best_idx = i
			}
		}

		
		if best_idx == -1 {
			break
		}

		
		tokens[best_idx] = best_id
		ordered_remove(&tokens, best_idx + 1)
	}

	if eos {
		append(&tokens, 2) 
	}

	return tokens[:]
}

