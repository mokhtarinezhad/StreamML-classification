


import os.path as op

defaultSep = ','

class Writer_abstract():

    def firstTracer(self):
        return

    def secondTracer(self):
        return

    def restart2traces(self):
        return

    def save(self): # for compatibility with SimpleWriter
        pass

    def set_name(self, g, **p): # for compatibility with SimpleWriter
        pass


class SimpleWriter(Writer_abstract):
    """
    Writing of printable objects without control on datas
    into a text file.
    usage:
    w = SimpleWriter('filename.txt', sep = ' ', mode = 'w')
    ...
    w.addline(s1, v1, ..., sn)
    with s1, v1, ... , printable objects
    Remark: This class could be simplified to print in Python 3.x
    """
    def __init__(self, file_name='', **kw):
        """Set the complete filename of a file to open in write mode
        @param file_name : StringType - the filename
        other param: see analParams.
        """
        self._analParams(**kw)
        if file_name:
            self.direct = op.dirname(file_name)
            self.fnbase = op.basename(file_name)
            self.filename = file_name
        else:
            self.filename = self.direct = self.fnbase = ''

    def _analParams(self, **kw):
        """
        @param kw: DictType - parameters with default values
        - sep : StringType - the separator of fields in the record.
        Default : ','. The separator support more than one character.
        e.g. : '","' or ')('
        @param mode : StringType - the writing mode of the file.
        Default: 'a' for appending
        """
        self.sep, self.mode = (kw.get(i,j) for i,j in \
        zip(['sep', 'mode'],[defaultSep, 'a']))
        # analyse the separator
        if len(self.sep) > 1: # check for instance '","', ')(' or "','"
            self._sep = self._sep1 # function
            self.endl = self.sep[0] + '\n'
        else:
            self._sep = self._sep0 # function
            self.endl = '\n'
        self.addline = self._openFile # function

    def set_name(self, file_name, **kw):
        """ Close the current file and
        set the filename of a new output file, without need to
        repeat the directory path.
        @param file_name : StringType - the filename
        other param: see analParams.
        """
        self.save()
        self._analParams(**kw)
        if file_name:
            self.fnbase = op.basename(file_name)
            direct = op.dirname(file_name)
            if direct:
                self.direct = direct
            self.filename = self.direct + self.fnbase

    def _sep0(self, s):
        self.writer = self.f.write

    def _sep1(self, s):
        self.f.write(s[-1])
        self.writer = self.f.write

    def _put_data(self, *data):
        """ Put printable objects into the file with the separators.
        @param data : list of printable objects forming a line
        """
        try:
            self.writer = self._sep
            for d in data:
                self.writer(self.sep)
                if isinstance(d, type(defaultSep)):
                    self.writer(d)
                else:
                    self.writer(str(d))
            self.writer(self.endl)
        except IOError:
            print(self.filename, ': writing error')
            self.save() # probably re-raise the exception

    def _openFile(self, *data):
        """
        open the object's file and write a new data line
        @param data : list of printable objects forming a line
        """
        try:
            self.f = open(self.filename, self.mode)
            self.addline = self._put_data
            self._put_data(*data)
        except IOError:
            raise IOError(self.filename)

    def save(self, mode = 'a'):
        if self.addline == self._put_data:
            self.f.close()
            self.mode = mode
            self.addline = self._openFile

    def reset(self):
        self.save('w')

    def __del__(self):
        self.save()
        del self # ???

tracer = SimpleWriter('trace.txt', mode = 'w')
writer = SimpleWriter('report.txt', mode = 'w')

class ConsoleWriter(Writer_abstract):
    """
    Print of printable objects with no control on datas
    onto the console.
    usage:
    w = ConsoleWriter('filename.txt', sep = ' ', mode = 'w')
    ...
    w.addline(s1, v1, ..., sn)
    with s1, v1, ... , printable objects
    Remark: This class could be obsolete in Python 3.x
    """
    def __init__(self, **kw):
        """
        @param kw: DictType - see _analParams.
        """
        self._analParams(**kw)
        self.line = ''

    def _analParams(self, **kw):
        """
        @param sep : StringType - the separator of fields in the record.
        Default : defaultSep.
        @param ncols : IntType - max number of character in a line.
        """
        self.sep, self.ncols = (kw.get(i,j) for i,j in \
        zip(['sep', 'ncols'],[defaultSep, 78]))
        # analyse the separator
        if len(self.sep) > 1: # check for instance '","' or "','"
            self._sep = self._sep1
            self.endl = '\b'+self.sep[0]
        else:
            self._sep = self._sep0
            self.endl = ''
        self.addline = self._print_data

    def printsep(self, item):
        self.line = self.line + item

    def _sep0(self, s):
        self.writer = self.printsep

    def _sep1(self, s):
        self.line = s[-1]
        self.writer = self.printsep

    def printAll(self):
        """Cut a string in slice of max. ncols characters
        and display the string eventually in several lines.
        """
        if len(self.line) > self.ncols:
            print(self.line[:self.ncols], end='')
            self.line = self.line[self.ncols:]
            if len(self.line):
                print()
                self.printAll()
        else:
            print(self.line, end='')
            self.line = ''

    def _print_data(self, *data):
        """ Put a list of strings or printable objects with the
        separators into a single string.
        @param dataList : ListType or printable object to print
        """
        self.writer = self._sep
        for d in data:
            self.writer(self.sep)
            if isinstance(d, type(defaultSep)):
                self.writer(d)
            else:
                self.writer(str(d))
        self.printAll()
        print(self.endl)

tty = ConsoleWriter(sep = ' ', ncols = 78)


class doubleTracer(Writer_abstract):
    """ Writing simutaneously on two writers
    Usage:
    obj = doubleTracer(traceprt, tracefile)
    ...
    obj.addline(value)

    if printing on the console is no more useful:
    obj.secondTracer()
    if printing on the console is again useful:
    obj.restart2traces()
    """
    def __init__(self, tracer1, tracer2):
        """ Get the two writer objects correctely initialised.
        @param tracer1: InstanceType - ConsoleWriter or SimpleWriter
        @param tracer2: InstanceType - SimpleWriter
        """
        self.t1 = tracer1
        self.t2 = tracer2
        self.addline = self._call2traces

    def _call2traces(self, *dataList):
        self.t1.addline(*dataList)
        self.t2.addline(*dataList)

    def firstTracer(self):
        self.addline = self.t1.addline

    def secondTracer(self):
        self.addline = self.t2.addline

    def restart2traces(self):
        self.addline = self._call2traces

    def save(self):
        self.t1.save()
        self.t2.save()

# test code

class TruckData:

    def __init__(self, timestamp, trNr, timeInd, timeInd_to_next):
        """
        @param timestamp : StringType - Original time string of the
        first signal of the truck.
        @param trNr : IntType - a truck number for easy identification
        @param timeInd : IntType - The index in the data base of the
        trigger time
        @param timeInd_to_next : IntType - The index in the data base
        of the next truck's trigger time (last index of this
        truck + 1).
        """
        self.trNr = trNr  # truck number
        self.timestamp = timestamp
        self.timeIndex = timeInd
        self.timeIndex_to_next = timeInd_to_next

    def __str__(self):
        return 'Truck %d at %s, (%d, %d)' % (self.trNr,\
        self.timestamp, self.timeIndex, self.timeIndex_to_next)

if __name__ == "__main__" :
    s = SimpleWriter('testWriting.txt', mode='w', sep='","')
    p = ConsoleWriter(sep=';')
    # the following loop controls the interface compatibility
    # between SimpleWriter and ConsoleWriter
    for f in (s, p):
        # Caution:  see fd = f.addline:  do not copy f.addline to
        # another variable before the first calling of f.addline !!!!!
        # fd will not be informed that f.addline is changed after
        # the first calling!
        fdopen = f.addline
        fdopen('col1', 'col2')
        fd = f.addline  # confort renaming
        fd('lng1', 50.0)
        fd('lng2', 30.0)
        fd(['lng2a', 45.0,12.9])
        fd(*['lng2b', 45.0,12.9])
        f.save()
        ### May be in another module if f is shared
        f.addline('lng3', 40.0)
        f.addline('lng4', 90.0)
        #renaming saves the previous file
        f.set_name('appending.txt', mode='w')
        f.addline('Titre1')
        f.addline(5.9887)
        f.save()
    t = TruckData('2014-07-24', 5, 0, 1000)
    f = doubleTracer(s, p)  # tracing in appending.txt
    f.addline('Double writing?')
    f.addline(1, 2, 5.8)
    f.firstTracer()
    f.addline('test1')
    f.restart2traces()
    f.addline(34)
    f.addline(t)
    f.addline('test4')
    f.save()
