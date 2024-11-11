#!/opt/anaconda3/envs/p11/bin/python3

import os
import sys
import re
import glob
import subprocess
import statistics
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from pyraf import iraf

import bottom_a



def flat_division(inlist):
    
    flats = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for i1 in 'jhk':
        flat_path = os.path.join(script_dir, 'lib', f'{i1}*.fits')
        flats[i1] = glob.glob(flat_path)
        
    for i2 in inlist:
        print(i2, '/', flats[0],  re.sub(r'.fits', r'_fl.fits', i2))
        bottom_a.imarith(i2, '/', flats[i2[0]],  re.sub(r'.fits', r'_fl.fits', i2))


def method2_1(fitslist):
    # skylevel subtraction
    levlist = [bottom_a.skystat(fitsname, 'median') for fitsname in fitslist]
    for index, f2 in enumerate(tqdm(fitslist, desc='{:<13}'.format('sub_skylevel'))):
        data, hdr = fits.getdata(f2, header=True)
        f3 = re.sub(r'.fits', r'_lev.fits', f2)
        #print(f'{f2} {levlist}')
        data = data.astype(float)
        data -= levlist[index]
        hdr['SKYCOUNT']=levlist[index]
        #print(f2, '- ', levlist[index], '\t=', f3)
        fits.writeto(f3, data, hdr, overwrite=True)

def method2_2(fitslist):
    levlist = [bottom_a.skystat(fitsname, 'median') for fitsname in fitslist]
    levmean = statistics.mean(levlist)
    levratio = [levmean/varr for varr in levlist]

    for index, fitsname in enumerate(tqdm(fitslist, desc='{:<13}'.format('div_skylevel'))):
        data, hdr = fits.getdata(fitsname, header=True)
        fitsname2 = re.sub(r'.fits', r'_ylev.fits', fitsname)
        data = data * levratio[index]
        hdr['SKYCOUNT']=levlist[index]
        #print(fitsname, '* ', levratio[index], '\t=', fitsname2)
        fits.writeto(fitsname2, data, hdr, overwrite=True)
    #2024/05/21 ああああああああああああ


def method3(flist, obnamelist):
    # make sky
    
    def self_sky(ondict):
        
        for key in tqdm(ondict, desc='{:<13}'.format('make_selfsky')):
            out = '_' + key + '_skyimg.fits'
            bottom_a.combine(ondict[key], out, 'median')

    def off_sky(firstoff, firston, offdict):

        for band in tqdm(firstoff, desc='{:<13}'.format('make_offsky')):
            off_hdulist = [bottom_a.readheader(fits) for fits in firstoff[band]]
            on_hdulist  = [bottom_a.readheader(fits) for fits in firston[band]]
            for hdu in on_hdulist:
                base = float(hdu.mjd)
                sky_hdu = min(off_hdulist, key=lambda x: abs(float(x.mjd) - base))
                out = '_' + band + '_' + hdu.object + '_skyimg.fits'
                bottom_a.combine(offdict[band+'_'+hdu.object], out, 'median', 'none')
                #print(out, 'was made')
        
    firstoff = {}
    firston  = {}
    offdict = {}
    ondict  = {}
    
    for index, obname in enumerate(obnamelist):
        band = flist[index][:5]
        
        if 'sky' in obname:
            if band+'_'+obname not in offdict:
                offdict[band+'_'+obname] = []
                if band not in firstoff:
                    firstoff[band] = []
                    
                firstoff[band].append(flist[index])
            offdict[band+'_'+obname].append(flist[index])
            
        else:
            if band+'_'+obname not in ondict:
                ondict[band+'_'+obname] = []
                if band not in firston:
                    firston[band] = []
                    
                firston[band].append(flist[index])
            ondict[band+'_'+obname].append(flist[index])

    if len(offdict) == 0:
        self_sky(ondict)
    else:
        off_sky(firstoff, firston, offdict)
        
    return [item for sublist in ondict.values() for item in sublist]



def method4(flist, obnamelist):
    # sky subtraction

    for i1 in tqdm(range(len(obnamelist)), desc='{:<13}'.format('sky_subtract')):
        band = flist[i1][:5]
        sky = '_' + band + '_' + obnamelist[i1] + '_skyimg.fits'
        out2 = re.sub(r'.fits', r'_sky.fits', flist[i1])
        bottom_a.imarith(flist[i1], '-', sky, out2)
        #print(flist[i1],'-', sky, '=' ,out2)




if __name__ == "__main__":

	print('flat_sky.py is a module.')