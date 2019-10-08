all: setup.py LoopsIntegration.so
setup.py:
	python ../get_setup.py -p LoopsIntegration.pyx -m LoopsIntegration
LoopsIntegration.so: LoopsIntegration.pyx
	python setup.py build_ext -i
install:
	mv LoopsIntegration.cpython-36m-x86_64-linux-gnu.so LoopsIntegration.so
	mv LoopsIntegration.so ../../../lib/
clean:
	rm -fr build LoopsIntegration.c *~ *.html
