#!/opt/anaconda3/envs/p11/bin/python3

import os
import sys
import re
import glob
import shutil
import subprocess
import signal
import numpy as np
from astropy.io import fits
from pyraf import iraf

#import download
import bottom_a
import flat_sky_a
import starmatch_a
import starmatch_a_ql
import comb_phot_a as com_p


def do_init():

	home_dir = os.path.expanduser("~")
		
	new_dir1 = os.path.join(home_dir, 'pipeline_datahub')
	new_dir2 = os.path.join(new_dir1, 'optcam')
	new_dir3 = os.path.join(new_dir1, 'kSIRIUS')
	new_dir4 = os.path.join(new_dir1, 'object')
	new_dir5 = os.path.join(new_dir1, 'workdir')
	new_dir6 = os.path.join(new_dir1, 'matrix')

	dirs_to_create = [new_dir1, new_dir2, new_dir3, new_dir4, new_dir5, new_dir6]
	
	for directory in dirs_to_create:
		if not os.path.exists(directory):
			os.makedirs(directory)
			print(f"Created directory: {directory}")
		else:
			print(f"Directory already exists: {directory}")

	path_program = os.path.abspath(__file__)
	dir_of_program = os.path.dirname(path_program)
	file_path = os.path.join(dir_of_program, 'main_a.param')

	paramlist = {
    'objfile_dir': f'{new_dir4}',
	'matrix_dir': f'{new_dir6}',
    'rawdata_infra': f'{os.path.join(new_dir3, "{date}")}',
    'rawdata_opt': f'{os.path.join(new_dir2, "{date}")}',
    'work_dir': f'{os.path.join(new_dir5, "{objectname}", "{date}")}'
	}

	with open(file_path, 'r') as file:
		lines = file.readlines()

	updated_lines = []
	for line in lines:
		try:
			param_name = line.split(maxsplit=1)[0]
		except:
			updated_lines.append(line)
			continue
		
		if param_name in paramlist:
			new_value = f"{paramlist[param_name]}"
			updated_line = f"{param_name:<16}'{new_value}'\n"
			updated_lines.append(updated_line)
		else:
			updated_lines.append(line)

	with open(file_path, 'w') as file:
		file.writelines(updated_lines)

	print(f"パラメータファイル {file_path} を更新しました。")
	sys.exit()


def rename_object(fitslist):

	uni_list = []
	edit = 0
	for band in fitslist:
		first_obname = 'None'
		for varr in fitslist[band]:
			hdu = fits.open(varr)
			obname = hdu[0].header['OBJECT']
			if obname != first_obname:
				first_obname = obname
				if obname in uni_list:
					edit = 1
					newname = f'{obname}_r'
					uni_list.append(newname)
				else:
					edit = 0
			if edit == 1:
				hdu[0].header['OBJECT'] = newname
				hdu.flush()
			hdu.close()


class readparam():

	def __init__(self, date='{date}', objectname='{objectname}'):
		
		path_program = os.path.abspath(__file__)
		dir_of_program = os.path.dirname(path_program)
		dir1 = os.path.join(dir_of_program, 'main_a.param')
		dir2 = os.path.join(dir_of_program, 'advanced_a.param')
		if not os.access(dir1, os.R_OK):
			print('main_a.param is not found')
			sys.exit()
		if not os.access(dir2, os.R_OK):
			print('advanced_a.param is not found')
			sys.exit()

		with open(dir1) as f1, open(dir2) as f2:
			lines = f1.readlines() + f2.readlines()
		for line in lines:
			if line.startswith('#') or line.strip() == '':
				continue
			varr = line.split()
			if len(varr)==1:
				print(f'param of {varr[0]} is none.')
				continue

			items = ''
			if varr[1].startswith(('\'', '\"', '(')):
#					varr[1] = varr[1].replace('\'', '', 1)
				for index, item in enumerate(varr[1:]):
					if item.endswith(('\'', '\"', ')')):
#							item = item.replace('\'', '', 1)
						items = items + item
						varr[1] = items
						break
					items = items + item + ' '
			
			varr[1] = varr[1].replace("{date}", date)
			varr[1] = varr[1].replace("{objectname}", objectname)
			exec(f'self.{varr[0]} = {varr[1]}')
				

class readobjfile():
	
	def __init__(self, param, objectname):
		
		jhkcoordslist = []
		gicoordslist = []

		dir1 = os.path.join(param.objfile_dir, objectname)
		if not os.access(dir1, os.R_OK):
			print(objectname, 'file is not found')
			sys.exit()

		with open(dir1, 'r') as f:
			section = None
			for index, line in enumerate(f):
				linelist = line.split()
				if line.startswith('#') or not linelist:
					continue
				if line.startswith('[') and line.endswith(']'):
					section = line[1:-1]
					continue
				if linelist[0] == 'SearchName':
					self.SearchName = []
					for parampart in linelist[1:]:
						if parampart.startswith('#'):
							break
						exec("self." + linelist[0] + ".append(" + parampart + ")")
					continue

				if len(linelist) == 1 or line[1].startswith('#'):
					continue
				elif section == 'Parameters':
					exec("self." + linelist[0] + " = " + linelist[1])
			
				if section == 'Coords_jhk':
					jhkcoordslist.append(f)

				if section == 'Coords_gi':
					gicoordslist.append(f)
		
		if jhkcoordslist:
			dir2 = os.path.join(param.work_dir, 'Reference_jhk.coo')
			with open(dir2, 'w') as f2:
				f2.write('#\n')
				for coo in jhkcoordslist:
					f2.write(coo)
					f2.write('\n')

		if gicoordslist:
			dir3 = os.path.join(param.work_dir, 'Reference_gi.coo')
			with open(dir3, 'w') as f3:
				f3.write('#\n')
				for coo in gicoordslist:
					f3.write(coo)
					f3.write('\n')


class readlog():
	
	def __init__(self, filename):
		
		if not os.access(filename, os.R_OK):
			print(filename,' is not found')
			self.log = 'nothing'
		
		else:
			self.log = 'exists'
			with open(filename) as f:
				for line in f:
					if line.startswith('#'):
						continue
					varr = line.split()
					if len(varr) <= 1 :
						continue
					exec("self." + varr[0] + " = " + varr[1])



class readheader():
    
    def __init__(self, fitslist):
        
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
        
        for f1 in fitslist:
            
            hdu = fits.open(f1)
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


def match_object(fitslist, search_name_list):
	
	obnamelist = []
	wrong_index = []
	for index, fits_file in enumerate(fitslist):
		try:
			HDUlist = fits.open(fits_file)
			obnamelist.append(HDUlist[0].header['OBJECT'])
		except:
			print('header is broken', fits_file)
			wrong_index.append(index)
	
	for index in wrong_index[::-1]:
		del fitslist[index]
		del obnamelist[index]
		
	outfitslist = []
	outobnamelist = []
	found_index = []
	for name1 in search_name_list:
		for index, name2 in enumerate(obnamelist):
			if name1 in name2:
				outfitslist.append(fitslist[index])
				outobnamelist.append(obnamelist[index])
				found_index.append(index)
		for index in found_index[::-1]:
			del fitslist[index]
			del obnamelist[index]
		found_index.clear()
			
	return outfitslist, outobnamelist
	

def glob_latestproc(bands, fitspro):
	
	fitslist = []
	key = '.fits'
	for i in fitspro[::-1]:
		key = '_' + i + key
	for band in bands:
		search_pattern = f'{band}*{key}'
		fitslist.extend(glob.glob(search_pattern))
	fitslist.sort()
	return fitslist


def glob_latestproc2(bands, fitspro):
	fitslist = {}
	key = '.fits'
	for i in fitspro[::-1]:
		key = '_' + i + key
	for band in bands:
		search_pattern = f'{band}*{key}'
		#print(f'search_pattern\n{search_pattern}')
		fitslist[band] = glob.glob(search_pattern)
		#print(f'fitslist\n{fitslist}')
		fitslist[band].sort()
		if not fitslist[band]:
			del fitslist[band]
	return fitslist
	
	

def write_log(filename, paramdict):
    
	# fitspro は中で展開可
	# paramdict = [param_name:param(str or list)]
	dict1 = {}
	linelist2 = []
	if os.access(filename, os.R_OK):
		with open(filename, 'r') as logr:
			linelist = logr.readlines()
		for line in linelist:
			if line.startswith('#') or not line.strip():
				linelist2.append(line)
			else:
				line0 = line.split('#', 1)
				list1 = line0[0].split()
				dict1[list1[0]] = list1[1]
				linelist2.append([list1[0], dict1[list1[0]], line0[1:]])

	for key in paramdict:
		if key in dict1:
			dict1[key] = paramdict[key]
		else:
			linelist2.append([key, paramdict[key]])

	with open(filename, 'w') as logw:
		for line in linelist2:
			if line.startswith('#') or not line.strip():
				logw.write(line + '\n')
			else:
				param_name = '{:<16}'.format(line[0])
				logw.write(param_name)
				if isinstance(line[1]):
					param_part = ''
					for varr in line[1]:
						param_part = param_part + varr
					else:
						param_part = line[1]
				logw.write(param_part)
				if len(line) > 3:
					logw.write('   #')
					logw.write(line[2:])
				logw.write('\n')



def make_objfile(objectname, param):

	objfile = os.path.join(param.objfile_dir, objectname)

	if os.access(objfile, os.R_OK):
		print(f'{objectname} file is already exists.')
		sys.exit()

	with open(objectname, 'w') as file:
		file.write('\n')
		file.write('[Parameters]\n')
		file.write('# Please add information of OBJECT.' + '\n')
		file.write('# Please separate names with a space.' + '\n')
		file.write('#  ex) \'Searchname1\' \'Searchname2\' \'Searchname3\'' + '\n\n')
		file.write('{:<16}'.format('ObjectName') + f'\'{objectname}\'' + '\n')
		file.write('{:<16}'.format('SearchName') + f'\'{objectname}\'' +'\n')
		file.write('\n\n')
		file.write('[Coords_jhk]\n')
		file.write('# Please add reference coordinates (in iraf format) for aperture phot.\n')
		file.write('# The order of coordinates is Obj → C1 → C2 → others.\n')
		file.write('# Please list as many coordinates as possible.\n')
		file.write('\n')
		file.write('[Coords_gi]\n')
		file.write('# Please add reference coordinates (in iraf format) for aperture phot.\n')
		file.write('# The order of coordinates is Obj → C1 → C2 → others.\n')
		file.write('# Please list as many coordinates as possible.\n')
		file.write('\n')
	
	print(f'{objectname} file is created')
		


def execute_code(param, objparam, log, bands=['haon_', 'haoff']):
	
	bands0 = ['haoff', 'haon']
	bottom_a.setparam()

	fitspro = []
	fitslist = {}
	obnamelist = {}

	print(f'start pipeline')
	print(f'object {argvs[1]}')

	if param.quicklook == 1:
		print('quicklook mode')


	if any(varr in bands0 for varr in ['haoff', 'haon']):
		try:
			iraf.chdir(param.rawdata_opt)
			globlist = glob_latestproc2(bands0, fitspro)
			#print(f'globlist\n{globlist}')
		except:
			#print(f'{param.rawdata_opt} is not exists.')
			globlist = []

	if globlist:
		os.makedirs(param.work_dir, exist_ok=True)
		subprocess.run(f'rm {param.work_dir}/*.fits', shell=True, stderr=subprocess.DEVNULL)
	else:
		print(f'rowdata not exists.')
		print(f'end')
		sys.exit()

	for band1 in globlist:
		if band1 in ['haon', 'haoff']:
			fitslist[band1], obnamelist[band1] = match_object(globlist[band1], objparam.SearchName)
			
	if fitslist:
		os.makedirs(param.work_dir, exist_ok=True)
		subprocess.run(f'rm {param.work_dir}/*.fits', shell=True, stderr=subprocess.DEVNULL)
	else:
		print(f'rowdata not exists.')
		print(f'end')
		sys.exit()


	for band1 in fitslist:
		if band1 == 'haon':
				for varr in fitslist[band1]:
					shutil.copy(varr, f'{param.work_dir}/haon_{varr[4:]}')
		else:
			for varr in fitslist[band1]:
					shutil.copy(varr, param.work_dir)
		

	iraf.chdir(param.work_dir)
	

	fitslist = glob_latestproc2(bands, fitspro)
	rename_object(fitslist)


	if param.flatdiv == 1:
		print('yetyetyet')
		sys.exit()
		fitslist = glob_latestproc(bands, fitspro)
		flat_sky_a.flat_division(fitslist)
		fitspro.append('fl')
	
	if param.cut == 1:
		fitslist = glob_latestproc2(bands, fitspro)
		bottom_a.cut(fitslist, param)
		fitspro.append('cut')

	if param.gflip == 1:
		fitslist = glob_latestproc2(bands, fitspro)
		if 'haoff' in fitslist:
			bottom_a.xflip(fitslist['haoff'])

	if param.sub_skylev == 1:
		fitslist = glob_latestproc(bands, fitspro)
		flat_sky_a.method2_1(fitslist)
		fitspro.append('lev')
	elif param.div_skylev == 1:
		fitslist = glob_latestproc(bands, fitspro)
		flat_sky_a.method2_2(fitslist)
		fitspro.append('ylev')
	elif param.sub_skylev == 1 and param.div_skylev == 1:
		fitslist = glob_latestproc(bands, fitspro)
		flat_sky_a.method2_1(fitslist)
		fitspro.append('lev')
	elif param.custom_skylev == 1:
		fitslist = glob_latestproc(bands, fitspro)
		print('custom_skylev yetyetyet')
		sys.exit()
		
	if param.skysub == 1:
		fitslist = glob_latestproc(bands, fitspro)
		header = readheader(fitslist)
		fitslist = flat_sky_a.method3(fitslist, header.object)
		header = readheader(fitslist)
		flat_sky_a.method4(fitslist, header.object)
		fitspro.append('sky')
	"""
	if param.starmatch == 1:
		fitslist = glob_latestproc2(bands, fitspro)
		starmatch_a.main(fitslist, param)
		fitspro.append('geo*')
	"""
	if param.starmatch == 1:
		if param.quicklook == 1:
			fitslist = glob_latestproc2(bands, fitspro)
			starmatch_a_ql.main(fitslist, param)
			fitspro.append('ql_geo*')
		else:
			fitslist = glob_latestproc2(bands, fitspro)
			starmatch_a.main(fitslist, param)
			fitspro.append('geo*')


	if param.comb_per_set == 1:
		fitslist = glob_latestproc2(bands, fitspro)
		com_p.comb_pset(fitslist)

	if param.comb_all == 1:
		fitslist = glob_latestproc2(bands, fitspro)
		com_p.comb_all(fitslist, argvs[2], argvs[1])

	if param.aperture_phot == 1:
		print('phot')


	if param.row_fits == 1:
		subprocess.run(f'rm {param.work_dir}/???????-????.fits', shell=True, stderr=subprocess.DEVNULL)
	if param.cut_fits == 1:
		subprocess.run(f'rm {param.work_dir}/*_cut.fits', shell=True, stderr=subprocess.DEVNULL)
	if param.lev_fits == 1:
		subprocess.run(f'rm {param.work_dir}/*_lev.fits', shell=True, stderr=subprocess.DEVNULL)
	if param.sky_fits == 1:
		subprocess.run(f'rm {param.work_dir}/*_sky.fits', shell=True, stderr=subprocess.DEVNULL)
	if param.geo_fits == 1:
		subprocess.run(f'rm {param.work_dir}/*_geo?.fits', shell=True, stderr=subprocess.DEVNULL)
	if param.coo_file == 1:
		subprocess.run(f'rm {param.work_dir}/*.coo', shell=True, stderr=subprocess.DEVNULL)
	if param.match_file == 1:
		subprocess.run(f'rm {param.work_dir}/*.match', shell=True, stderr=subprocess.DEVNULL)
	if param.geo_file == 1:
		subprocess.run(f'rm {param.work_dir}/*.geo', shell=True, stderr=subprocess.DEVNULL)

	print("end")



if __name__ == '__main__':
	
	argvs = sys.argv
	argc = len(argvs)
	fitspro = []
	
	if argc == 1:
		print('usage1 ./ku1mV.py [OBJECT file name] ')
		print('usage2 ./ku1mV.py [object name][YYMMDD]')
		sys.exit()

	if argvs[1] == 'init':
		print('init')
		do_init()

	if argc == 2:
		#object ファイルの生成
		param = readparam('', argvs[1])
		os.makedirs(param.objfile_dir, exist_ok=True)
		iraf.chdir(param.objfile_dir)
		make_objfile(argvs[1], param)
	
	elif argc == 3:
		param = readparam(argvs[2], argvs[1])
		objparam = readobjfile(param, argvs[1])
		execute_code(param, objparam, None)

	elif argc == 4:
		print('yetyetyet, only jhk')
		sys.exit()
		param = readparam(argvs[2], argvs[1])
		objparam = readobjfile(param, argvs[1])

		path = os.path.join(param.work_dir)
		os.makedirs(path, exist_ok=True)
		iraf.chdir(path)
		log = readlog('log.txt')
		execute_code(param, objparam, log, argvs[3])
	
	else:
		print('usage1 ./Ha_main.py [OBJECT file name] ')
		print('usage2 ./Ha_main.py [object name][YYMMDD]')