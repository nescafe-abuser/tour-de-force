default: build

build:
  odin build .

run: build
  ./tour_de_force

clean: 
  rm tour_de_force

init:
  chmod +x get.sh
  ./get.sh
