.PHONY: all
all: python

.PHONY: configure
configure: build

.PHONY: python
python: mcexercise.so python.py
	./python.py

.PHONY: so
so: mcexercise.so
mcexercise.so: build
	cmake --build build

build: main.cpp CMakeLists.txt voro/2d/CMakeLists.txt
	cmake -S. -Bbuild

voro/2d/CMakeLists.txt:
	git submodule update --init
