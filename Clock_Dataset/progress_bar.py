# Full progress bar during long loop:
def progress_bar(loop_length, running_idx, bar_length = 20, nProg=0):

    import sys

    nWLs = loop_length  # how many steps in the loop

    # Progress Bar setup:
    ProgMax = bar_length    # number of dots in progress bar
    if nWLs<ProgMax:   ProgMax = nWLs   # if less than 20 points in scan, shorten bar
    print ("|" +    ProgMax*"-"    + "|     MyFunction() progress")
    sys.stdout.write('|'); sys.stdout.flush();  # print start of progress bar
    #nProg = 0   # fraction of progress

    # update progress bar:
    if ( running_idx >= nProg*nWLs/ProgMax ):
        '''Print dot at some fraction of the loop.'''
        sys.stdout.write('*'); sys.stdout.flush();  
        nProg = nProg+1
        return nProg
    if ( running_idx >= nWLs-1 ):
        '''If done, write the end of the progress bar'''
        sys.stdout.write('|     done  \n'); sys.stdout.flush(); 
        return nProg