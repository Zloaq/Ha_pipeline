#!/opt/anaconda3/envs/p11/bin/python3

import os
import sys
import re
from tqdm import tqdm

from astropy.io import fits

import bottom_a
#import dostack_auto



def comb_pset(fitslist):
    #自分の好きなobject の名前を反映させる。2024/06/12
    sep_bandset = {}
    for band in fitslist:
        header = bottom_a.readheader(fitslist[band])
        for filename, obname in zip(header.filename, header.object):
            if f'{filename[:11]}_{obname}' not in sep_bandset:
                sep_bandset[f'{filename[:11]}-{obname}'] = []
            sep_bandset[f'{filename[:11]}-{obname}'].append(filename)

    for nameid in tqdm(sep_bandset, desc='{:<13}'.format("combine")):
        #print(f'{nameid}\n{sep_bandset[nameid]}')
        outname = f'{nameid}.fits'
        bottom_a.combine(sep_bandset[nameid], outname, 'average')


def comb_all(fitslist, day, obname):

    for band, varrlist in tqdm(fitslist.items(), desc='{:<13}'.format("comb all")):
        outname = f'{band}{day}_{obname}.fits'
        bottom_a.combine(varrlist, outname, 'average')

"""
def match_star_coords(fitslist, refcoofile, fwhms=[3, 4, 5]):
    print('match')
    # Reference_bands.coo
    # daofind のために半値幅を選定
    # phot のための半値幅
    for fitsname in fitslist:
        data = fits.getdata(fitsname)
        nxblock = len(data[0])
        nyblock = len(data)
        starnum = {}
        for fwhm in fwhms:
            threshold = 5
            while threshold > 3:
                outputf = re.sub('.fits', f'{fwhm}.coo', fitsname)
                os.rm(outputf)
                bottom_a.daofind(fitsname, outputf, fwhm, threshold)
                dictlist = dostack_auto.daofind_result([outputf], [fitsname], nxblock, nyblock)
                if len(dictlist['xcen']) < 3:
                    threshold -= 0.5
                    continue
                starnum[fwhm] = len(dictlist['xcen'])
                break
            if not starnum[fwhm]:
                print(f'no star {fitsname} fwhm={fwhm}')
                break
"""


def phot(fitslist, coordslist, fwhm):
    print('phot')

def method1():
    print('method1')

if __name__ == '__main__':
    print('mosule')