all: setup.py LoopsIntegration.so
setup.py:
	get_setup.py -p LoopsIntegration.pyx -m LoopsIntegration
LoopsIntegration.so: LoopsIntegration.pyx
	python3 setup.py build_ext -i
	#mv LoopsIntegration.cpython-37m-darwin.so LoopsIntegration.so
install:
	mv LoopsIntegration.so ../../../lib/
clean:
	rm -fr build LoopsIntegration.c *~ *.html
