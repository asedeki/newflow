all: setup.py cinteraction.so
setup.py:
	python ../get_setup.py -p cinteraction.pyx -m cinteraction
cinteraction.so: cinteraction.pyx 
	python setup.py build_ext -i
	#mv cinteraction.cpython-37m-darwin.so cinteraction.so
install:
	mv cinteraction.cpython-36m-x86_64-linux-gnu.so cinteraction.so
	mv cinteraction.so ../../../lib/
clean:
	rm -fr build cinteraction.c *~ *.html
