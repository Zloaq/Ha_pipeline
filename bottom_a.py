#!/opt/anaconda3/envs/p11/bin/python3

import os
import re
import sys
import math
from tqdm import tqdm
import scipy as sp
import numpy as np
from astropy.io import fits
from scipy.stats import mode
from pyraf import iraf


class readheader():
    
    def __init__(self, inlist):

        self.filename = []
        self.object = []
        self.exptime = []
        self.jd = []
        self.mjd = []
        self.ra = []
        self.dec = []
        self.airmass = []
        self.offsetra = []
        self.offsetde = []
        self.offsetro = []
        self.azimuth = []
        self.altitude = []
        self.rotator = []
        
        for f1 in inlist:
            
            hdu = fits.open(f1)
            self.filename.append(f1)
            self.object.append(hdu[0].header['OBJECT'])
            self.exptime.append(hdu[0].header.get('EXPTIME') or hdu[0].header.get('EXP_TIME'))
            self.jd.append(hdu[0].header['JD'])
            self.mjd.append(hdu[0].header['MJD'])
            self.ra.append(hdu[0].header['RA'])
            self.dec.append(hdu[0].header['DEC'])
            self.airmass.append(hdu[0].header['AIRMASS'])
            self.offsetra.append(hdu[0].header['OFFSETRA'])
            self.offsetde.append(hdu[0].header['OFFSETDE'])
            self.offsetro.append(hdu[0].header['OFFSETRO'])
            self.azimuth.append(hdu[0].header['AZIMUTH'])
            self.altitude.append(hdu[0].header['ALTITUDE'])
            self.rotator.append(hdu[0].header['ROTATOR'])
            hdu.close()


def setparam(band='g'):
	
    scale = {
         'g': 0.692, 'i': 0.692, 'j': 0.692, 'h': 0.692, 'k': 0.692
    }
    readnoise = {
        'g': 11.97, 'i': 11.97, 'j': 11.97, 'h': 11.97, 'k': 11.97
    }
    epadu = {
        'g': 11.97, 'i': 11.97, 'j': 10, 'h': 11.7, 'k': 9.5
    }

    iraf.noao()
    iraf.digi(Stdout=1)
    iraf.app(Stdout=1)
    iraf.images()
    iraf.immatch()
    iraf.tv()
    iraf.digiphot.unlearn()
    iraf.apphot.unlearn()
    iraf.ptools.unlearn()
    iraf.txdump.unlearn()
    iraf.apphot.datapars.scale = scale[band]
    iraf.apphot.datapars.fwhmpsf = 12
    iraf.apphot.datapars.datamax = 15000
    iraf.apphot.datapars.datamin = 0
    iraf.apphot.datapars.readnoise = readnoise[band]
    iraf.apphot.datapars.epadu = epadu[band]
    iraf.apphot.centerpars.calgorithm = 'centroid'
    iraf.apphot.centerpars.cbox = 3
    iraf.apphot.centerpars.minsnratio = 2
    iraf.apphot.centerpars.maxshift = 1
    iraf.apphot.fitskypars.salgorithm = 'median'
    iraf.apphot.fitskypars.dannulus = 10
    iraf.apphot.findpars.threshold = 5
    iraf.apphot.findpars.sharphi = 0.8


def cut(inlist, param):
    cutrange = {
        'haoff': param.haoff_cutrange, 'haon_': param.haon__cutrange,
    }
    in_filenames = []
    out_filenames = []
    crange_list = []
    
    for f0 in inlist:
        for f1 in inlist[f0]:
            crange_list.append(cutrange[f0])
            f2 = re.sub('.fits', '_cut.fits', f1)
            in_filenames.append(f1)
            out_filenames.append(f2)
            #slices = [slice(*map(int, item.split(':'))) for item in crange_list.strip('[]').split(', ')]
    
    for f1, f2, cr in tqdm(zip(in_filenames, out_filenames, crange_list), desc='{:<13}'.format("imcopy"), total=len(in_filenames)):
        with fits.open(f1) as hdul:
            data = hdul[0].data 
            header = hdul[0].header 
            exec(f'data = data{cr}') 
            hdu = fits.PrimaryHDU(data, header) 
            hdu.writeto(f2, overwrite=True) 


def xflip(fitslist):
    for varr in tqdm(fitslist, desc='{:<13}'.format(f'xflip')):
        data, hdr = fits.getdata(varr, header=True)
        data = np.flip(data, axis=1)
        fits.writeto(varr, data, hdr, overwrite=True)



def skystat(fitsname, method, nclip=3, clip_threshold=3):
    
    data = fits.getdata(fitsname)
    flat_data = data.flatten()
    threshold = np.percentile(flat_data, 100)
    backgroud_data = flat_data[flat_data <= threshold]
    mode_value, count = mode(backgroud_data)
    median_value = np.median(backgroud_data)

    if method == 'mode':
        return mode_value
    if method == 'mode,stddev':
        basis_value = mode_value
    
    for num in range(nclip):
        
        if 'mode' in method:
            basis_value, count = mode(backgroud_data)
        else:
            basis_value = np.median(backgroud_data)
        
        squared_diffs = (backgroud_data - basis_value) ** 2
        mean_squared_diff = np.mean(squared_diffs)
        rms = np.sqrt(mean_squared_diff)
        lower_bound = basis_value - clip_threshold * rms
        upper_bound = basis_value + clip_threshold * rms
        backgroud_data = backgroud_data[(backgroud_data >= lower_bound) & (backgroud_data <= upper_bound)]
    
    skyrms = rms
    backgroud_data = backgroud_data[(backgroud_data <= basis_value)]
    squared_diffs = (backgroud_data - basis_value) ** 2
    mean_squared_diff = np.mean(squared_diffs)
    lowside_rms = np.sqrt(mean_squared_diff)


    if method == 'stddev':
        return lowside_rms
    elif method == 'median':
        return basis_value
    elif method == 'mode,stddev':
        return basis_value, skyrms
    elif method == 'median,stddev':
        return basis_value, lowside_rms


def combine(inlist, outname, method, reject='sigclip'):
	
    with open('_inlist', 'w')as f1:
        for i1 in inlist:
            f1.write(i1+'\n')
        
    iraf.imcombine.unlearn()
    iraf.imcombine.combine = method
    iraf.imcombine.reject = reject
    iraf.imcombine.zero = 'none'
    iraf.imcombine.mclip = 'yes'
    iraf.imcombine.lsigma = 3.0
    iraf.imcombine.hsigma = 3.0
    
    iraf.imcombine('@_inlist', outname, Stdout=1)


def imarith(infile1, op, infile2,  outfile):

    if op != '+' and op != '-' and op != '/':
            print("bad operator, op")
            return 1

    data1, hdr = fits.getdata(infile1, header=True)
    data2 = fits.getdata(infile2)

    if op == '+':
            data3 = data1 + data2
    elif op == '-':
            data3 = data1 - data2
    elif op == '/':
            data3 = data1 / data2

    fits.writeto(outfile, data3, hdr, overwrite=True)


def center(inputfits, outputf, inputcoords='temp.coo'):
    if os.path.exists('temp1.coo'):
        os.remove('temp1.coo')
    mode, stddev = skystat(inputfits, 'mode,stddev')
    iraf.apphot.datapars.sigma = stddev
    iraf.apphot.datapars.datamin = float(mode)+float(stddev)
    iraf.apphot.center.unlearn()
    iraf.apphot.centerpars.calgorithm = 'centroid'
    iraf.apphot.center.coords = inputcoords
    iraf.apphot.center.output = 'temp1.coo'
    iraf.apphot.center.interactive = 'no'
    iraf.apphot.center.verify = 'no'
    iraf.apphot.center.update = 'yes'
    iraf.apphot.center.verbose = 'no'
    iraf.apphot.center.mode = 'h'
    iraf.apphot.center(inputfits)

    if not os.path.exists('temp1.coo'):
        #print(f'temp1.coo is not exists {inputfits}')
        file2 = open(outputf, 'w')
        file2.close()
        os.remove(inputcoords)
        return
    with open('temp1.coo', 'r') as file1:
        lines = file1.readlines()
    filtered_lines = [line for line in lines if not inputcoords in line]
    with open(outputf, 'w') as file2:
        file2.writelines(filtered_lines)

    


def daofind(inputf, outputf, fwhm, threshold=5 ):
    
    iraf.noao()
    iraf.digiphot(Stdout=1)
    iraf.apphot(Stdout=1)

    mode, stddev = skystat(inputf, 'mode,stddev')

    iraf.apphot.datapars.scale = 0.692
    iraf.apphot.datapars.fwhmpsf = fwhm
    iraf.apphot.datapars.sigma = stddev
    iraf.apphot.datapars.datamin = float(mode)-float(stddev)
    iraf.apphot.datapars.datamax = 15000
    iraf.apphot.findpars.threshold = threshold
    iraf.daofind.unlearn()
    iraf.daofind.output = 'temp1.coo'
    iraf.daofind.interac = 'no'
    iraf.daofind.verify = 'no'
    iraf.daofind.mode = 'h'
    iraf.daofind(inputf)

    iraf.apphot.center.unlearn()
    iraf.apphot.centerpars.calgorithm = 'centroid'
    iraf.apphot.center.coords = 'temp1.coo'
    iraf.apphot.center.output = 'temp2.coo'
    iraf.apphot.center.interactive = 'no'
    iraf.apphot.center.verify = 'no'
    iraf.apphot.center.update = 'yes'
    iraf.apphot.center.verbose = 'no'
    iraf.apphot.center.mode = 'h'
    iraf.apphot.center(inputf)

    if not os.path.exists('temp2.coo'):
        print(f'temp2.coo is not exists')
        file2 = open(outputf, 'w')
        file2.close()
        os.remove('temp1.coo')
        return

    with open('temp2.coo', 'r') as file1:
        lines = file1.readlines()
    filtered_lines = [line for line in lines if not "temp1.coo" in line]
    with open(outputf, 'w') as file2:
        file2.writelines(filtered_lines)

    os.remove('temp1.coo')
    os.remove('temp2.coo')



def psfmeasure(inputf):
	
    iraf.noao()
    iraf.obsutil(Stdout=1)
    iraf.psfmeasure.unlearn()
    iraf.psfmeasure.coords = 'center'
    iraf.psfmeasure.scale = 0.69
    iraf.psfmeasure.display = 'No'
    iraf.psfmeasure.logfile = ''
    iraf.psfmeasure.mode = 'h'
    
    return iraf.psfmeasure(inputf, Stdout=1)



def imshift(inputf, outputf, xshift, yshift):
	
	iraf.imshift.mode = 'h'
	iraf.imshift(inputf, outputf, xshift, yshift)


def xyxymatch(inputf, referencef, outputf, torelance):
    
    iraf.xyxymatch.unlearn()
    iraf.xyxymatch.input = inputf
    iraf.xyxymatch.reference = referencef
    iraf.xyxymatch.output = outputf
    iraf.xyxymatch.tolerance = torelance
    iraf.xyxymatch.separation = 3
    iraf.xyxymatch.matching = 'triangles'
    iraf.xyxymatch.nmatch = 40
    iraf.xyxymatch.ratio = 10
    iraf.xyxymatch.interactive = 'No'
    iraf.xyxymatch.verbose = 'Yes'
    iraf.xyxymatch.mode = 'h'
    iraf.xyxymatch(inputf, Stdout=1)


def geomap(inputf, xmaxv, ymaxv, outputf, fitmetry, func='polynomial'):
    
    iraf.geomap.unlearn()
    iraf.geomap.input = inputf
    iraf.geomap.database = outputf
    iraf.geomap.xmin = 'INDEF'
    iraf.geomap.xmax = 'INDEF'
    iraf.geomap.ymin = 'INDEF'
    iraf.geomap.ymax = 'INDEF'
    iraf.geomap.fitgeometry = fitmetry
    iraf.geomap.function = func
    iraf.geomap.maxiter = 10
    iraf.geomap.interactive = 'No'
    iraf.geomap.mode = 'h'
    iraf.geomap(inputf, outputf, Stdout=1)



def geotran_param(xmaxv='INDEF', ymaxv='INDEF'):
    
    iraf.geotran.unlearn()
    iraf.geotran.database = ''
    iraf.geotran.transforms = ''
    iraf.geotran.geometry = 'linear'
    iraf.geotran.xmag = 1
    iraf.geotran.ymag = 1
    iraf.geotran.xmin = 'INDEF'
    iraf.geotran.xmax = 'INDEF'
    iraf.geotran.ymin = 'INDEF'
    iraf.geotran.ymax = 'INDEF'
    iraf.geotran.xscale = 1
    iraf.geotran.yscale = 1
    iraf.geotran.interpo = 'drizzle'
    iraf.geotran.boundary = 'constant'
    iraf.geotran.boundary = 'nearest'
    iraf.geotran.constant = -100000
    iraf.geotran.fluxconserve = 'yes'
    iraf.geotran.verbose = 'no'
    iraf.geotran.mode = 'h'


def geotran(inf, outf, xmean, ymean, xrefmean, yrefmean, xrotation, yrotation):


    iraf.geotran.xin = xmean
    iraf.geotran.yin = ymean
    iraf.geotran.xout = xrefmean
    iraf.geotran.yout = yrefmean
    iraf.geotran.xrotation = xrotation
    iraf.geotran.yrotation = yrotation
    iraf.geotran.mode = 'h'
    iraf.geotran(inf, outf)


def phot(fitsname, fwhm, sigma):

    iraf.imstatistics.unlearn()
    iraf.imstatistics.fields = 'mode,stddev'
    iraf.imstatistics.nclip = 5
    iraf.imstatistics.format = 'No'
    varr = iraf.imstatistics(fitsname, Stdout=1)
    varr = varr[0].split()
    mode = varr[0]
    stddev = varr[1]
    
    iraf.apphot.datapars.fwhmpsf = fwhm
    iraf.apphot.datapars.sigma = sigma
    iraf.apphot.datapars.datamin = float(mode)-2*float(stddev)
    iraf.apphot.datapars.datamax = 15000
    iraf.apphot.datapars.readnoise = 10
    iraf.apphot.datapars.epadu = 1
    #まだだだだだだ

    iraf.apphot.fitskypars.salgorithm = 'mode'
    iraf.apphot.fitskypars.annulus = 2.5*float(fwhm)
    iraf.apphot.fitskypars.dannulus = 10
    iraf.apphot.photpars.apertures = fwhm
    iraf.apphot.photpars.zmag = 0

    iraf.apphot.phot.interactive = 'no'
    iraf.apphot.phot.cache = 'no'
    iraf.apphot.phot.verify = 'no'
    iraf.apphot.phot.update = 'yes'
    iraf.apphot.phot.verbose = 'no'
    iraf.apphot.phot.mode = 'h'




if __name__ == '__main__':
	
	print('bottom_a.py is a module.')
