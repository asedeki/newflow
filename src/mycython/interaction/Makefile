all: cinteraction.so
cinteraction.so: cinteraction.pyx 
	python3 setup.py build_ext -i
	mv cinteraction.cpython-37m-darwin.so cinteraction.so
install:
	mv cinteraction.so ../../../lib/
clean:
	rm -fr build cinteraction.c *~ *.html
