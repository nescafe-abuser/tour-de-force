default: build

build:
  odin build .

run: build
  ./tour-de-force -c=stories15M.bin -t=tokenizer.bin

clean: 
  rm tour-de-force

init:
  chmod +x get.sh
  ./get.sh
