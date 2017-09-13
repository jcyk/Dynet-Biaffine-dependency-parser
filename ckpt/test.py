import os

rootdir = '.'
for fil in os.listdir(rootdir):
	path = os.path.join(rootdir, fil)
	if os.path.isdir(path):
		fname = os.path.join(path, 'best_test')
		if os.path.exists(fname):
			os.system('perl ../run/eval.pl -q -b -g %s -s %s -o tmp' % (fname, '../../sancl_data/norm_data/emails-norm-test.conll'))
			os.system('tail -n 3 tmp > score_tmp')
			LAS, UAS = [float(line.strip().split()[-2]) for line in open('score_tmp').readlines()[:2]]
			print path
			print 'LAS %.2f, UAS %.2f'%(LAS, UAS)
			os.system('rm tmp score_tmp')
