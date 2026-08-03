package tdf

import "core:flags"
import "core:fmt"
import "core:os"

Options :: struct {
	checkpoint:    string `args:"name=c,required" usage:"Path to binary model checkpoint file."`,
	tokenizer:     string `args:"name=t,required" usage:"Path to binary tokenizer file."`,
	temperature:   f32 `args:"name=temp"       usage:"Sampling temperature (0.0 = greedy). [default: 0.9]"`,
	topp:          f32 `args:"name=topp"       usage:"Top-P nucleus sampling threshold. [default: 0.9]"`,
	steps:         int `args:"name=n"          usage:"Max number of tokens to generate. [default: 256]"`,
	prompt:        string `args:"name=p"          usage:"Input prompt string."`,
	system_prompt: string `args:"name=sys"        usage:"Optional system prompt for chat mode."`,
	mode:          string `args:"name=mode"       usage:"Execution mode: 'generate' or 'chat'. [default: generate]"`,
	seed:          u64 `args:"name=seed"       usage:"RNG seed for sampling. [default: 0]"`,
}

main :: proc() {
	opts := Options {
		temperature = 0.9,
		topp        = 0.9,
		steps       = 256,
		mode        = "generate",
		seed        = 0,
	}

	flags.parse_or_exit(&opts, os.args)

	fmt.println("Loading checkpoint:", opts.checkpoint)
	transformer, model_ok := init_transformer(opts.checkpoint)
	if !model_ok {
		fmt.eprintln("Failed to initialize model architecture.")
		os.exit(1)
	}
	defer free_transformer(&transformer)

	fmt.println("Loading tokenizer:", opts.tokenizer)
	tokenizer, tok_ok := init_tokenizer(opts.tokenizer, transformer.config.vocab_size)
	if !tok_ok {
		fmt.eprintln("Failed to initialize tokenizer.")
		os.exit(1)
	}
	defer free_tokenizer(&tokenizer)

	sampler := init_sampler(transformer.config.vocab_size, opts.temperature, opts.topp, opts.seed)
	defer free_sampler(&sampler)

	fmt.println("--- Inference Engine Initialized ---")

	switch opts.mode {
	case "generate":
		generate(&transformer, &tokenizer, &sampler, opts.prompt, opts.steps)
	case "chat":
		chat(&transformer, &tokenizer, &sampler, opts.prompt, opts.system_prompt, opts.steps)
	case:
		fmt.eprintfln("Unknown mode '%s'. Valid modes are: 'generate', 'chat'", opts.mode)
		os.exit(1)
	}
}

