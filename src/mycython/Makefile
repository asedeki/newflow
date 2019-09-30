all: bul int
bul: loops/LoopsIntegration.pyx
	cd loops;make;make install;make clean
int: interaction/cinteraction.pyx
	cd interaction/;make
install:
	cd loops;make install;make clean
	cd interaction/;make install; make clean
