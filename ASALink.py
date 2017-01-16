from model import *
import time
import numpy as np
import os



#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################







# THE PROBLEM RIGHT NOW IS THAT WE HAVE TO FORCE VALUES TO BE ZERO FOR 
# PEOPLE THAT WE KNOW THE SEX OF!!!!!!!!!!!!!!!!!!!!!!







#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################




def personStruct(IP):
	
	if(IP == 'AD' or IP == 'AR'):
		ans = '\
	struct Person {\n\
		Person *parent1;\n\
		Person *parent2;\n\
		double phi1;\n\
		double phi2;\n\
		double p1;\n\
		double p2;\n\
		double p3;\n\
		void updateCartesian() {\n\
			this->p1 = sin(this->phi1)*sin(this->phi1)*sin(this->phi2)*sin(this->phi2);\n\
			this->p2 = sin(this->phi1)*sin(this->phi1)*cos(this->phi2)*cos(this->phi2);\n\
			this->p3 = cos(this->phi1)*cos(this->phi1);\n\
		}\
		};'
	elif(IP == 'XL'):
		ans = '\
		struct Person {\n\
		double p1;\n\
		double p2;\n\
		double p3;\n\
		double p4;\n\
		double p5;\n\
		};\n\n'
		ans_male = '\
	struct PersonMale : Person {\n\
		Person *parent1;\n\
		Person *parent2;\n\
		double phi1;\n\
		void updateCartesian() {\n\
			this->p1 = 0;\n\
			this->p2 = 0;\n\
			this->p3 = 0;\n\
			this->p4 = sin(this->phi1)*sin(this->phi1);\n\
			this->p5 = cos(this->phi1)*cos(this->phi1);\n\
		}};\n\n'
		ans_female = '\
	struct PersonFemale : Person {\n\
		Person *parent1;\n\
		Person *parent2;\n\
		double phi1;\n\
		double phi2;\n\
		void updateCartesian() {\n\
			this->p1 = sin(this->phi1)*sin(this->phi1)*sin(this->phi2)*sin(this->phi2);\n\
			this->p2 = sin(this->phi1)*sin(this->phi1)*cos(this->phi2)*cos(this->phi2);\n\
			this->p3 = cos(this->phi1)*cos(this->phi1);\n\
			this->p4 = 0;\n\
			this->p5 = 0;\n\
		}};\n\n'
		ans_unknown = '\
	struct PersonUnknown : Person {\n\
		Person *parent1;\n\
		Person *parent2;\n\
		double phi1;\n\
		double phi2;\n\
		double phi3;\n\
		double phi4;\n\
		void updateCartesian() {\n\
			this->p1 = sin(this->phi1)*sin(this->phi1)*sin(this->phi2)*sin(this->phi2)*sin(this->phi3)*sin(this->phi3)*sin(this->phi4)*sin(this->phi4);\n\
			this->p2 = sin(this->phi1)*sin(this->phi1)*sin(this->phi2)*sin(this->phi2)*sin(this->phi3)*sin(this->phi3)*cos(this->phi4)*cos(this->phi4);\n\
			this->p3 = sin(this->phi1)*sin(this->phi1)*sin(this->phi2)*sin(this->phi2)*cos(this->phi3)*cos(this->phi3);\n\
			this->p4 = sin(this->phi1)*sin(this->phi1)*cos(this->phi2)*cos(this->phi2);\n\
			this->p5 = cos(this->phi1)*cos(this->phi1);\n\
		}};\n\n'
		ans += ans_male+ans_female+ans_unknown
	else:
		assert 0,'This inheritance pattern isn\'t implemented yet!'
	return ans


def initPerson(person,startIndex,IP):

	name = 'person_'+(str(person.Id).replace('-','_'))
	if(IP == 'AD' or IP == 'AR'):
		ans = '\tPerson '+name+';\n'
	elif(IP == 'XL'):
		if(person.sex == 'male'):
			ans = '\tPersonMale '+name+';\n'
		elif(person.sex == 'female'):
			ans = '\tPersonFemale '+name+';\n'
		elif(person.sex == 'unknown'):
			ans = '\tPersonUnknown '+name+';\n'
		else:
			assert 0,'Invalid sex'
	else:
		assert 0,'Not implemented'

	# write out the parents
	if(len(person.parents) > 0):
		parent1Name = 'person_'+(str(person.parents[0].Id).replace('-','_'))
		parent2Name = 'person_'+(str(person.parents[1].Id).replace('-','_'))	
		ans += '\t'+name+'.parent1 = &'+parent1Name+';\n'
		ans += '\t'+name+'.parent2 = &'+parent2Name+';\n'

	# write out their probabilities
	if(len(person.parents) == 0):
		if(IP == 'AD' or IP == 'AR'):
			ans += '\t'+name+'.phi1 = x['+str(startIndex[person])+'];\n'
			ans += '\t'+name+'.phi2 = x['+str(startIndex[person]+1)+'];\n'
			ans += '\t'+name+'.updateCartesian();\n'
		elif(IP == 'XL'):
			if(person.sex == 'male'):
				ans += '\t'+name+'.phi1 = x['+str(startIndex[person])+'];\n'
			elif(person.sex == 'female'):
				ans += '\t'+name+'.phi1 = x['+str(startIndex[person])+'];\n'
				ans += '\t'+name+'.phi2 = x['+str(startIndex[person]+1)+'];\n'
			elif(person.sex == 'unknown'):
				ans += '\t'+name+'.phi1 = x['+str(startIndex[person])+'];\n'
				ans += '\t'+name+'.phi2 = x['+str(startIndex[person]+1)+'];\n'
				ans += '\t'+name+'.phi3 = x['+str(startIndex[person]+2)+'];\n'
				ans += '\t'+name+'.phi4 = x['+str(startIndex[person]+3)+'];\n'
			else:
				assert 0,'Invalid sex'
			
			ans += '\t'+name+'.updateCartesian();\n'
		else:
			assert 0,'This inheritance pattern isn\'t implemented yet!'
	else:
		if(IP == 'AD' or IP == 'AR'):
			ans += '\t'+name+'.p1 = '+name+'.parent1->p1*(1.0*'+name+'.parent2->p1+0.5*'+name+'.parent2->p2)+'+name+'.parent1->p2*(0.5*'+name+'.parent2->p1+0.25*'+name+'.parent2->p2);\n'
			ans += '\t'+name+'.p2 = '+name+'.parent1->p1*(0.5*'+name+'.parent2->p2+1.0*'+name+'.parent2->p3)+'+name+'.parent1->p2*(0.5*'+name+'.parent2->p1+0.5*'+name+'.parent2->p2+0.5*'+name+'.parent2->p3)+'+name+'.parent1->p3*(1.0*'+name+'.parent2->p1+0.5*'+name+'.parent2->p2);\n'
			ans += '\t'+name+'.p3 = '+name+'.parent1->p2*(0.25*'+name+'.parent2->p2+0.5*'+name+'.parent2->p3)+'+name+'.parent1->p3*(0.5*'+name+'.parent2->p2+1.0*'+name+'.parent2->p3);\n'
		elif(IP == 'XL'):
			if(person.parents[0].sex == 'female' and person.parents[1].sex == 'male'):
				if(person.sex == 'male'):
					ans += '\t'+name+'.p1 = 0.;\n'
					ans += '\t'+name+'.p2 = 0.;\n'
					ans += '\t'+name+'.p3 = 0.;\n'
					ans += '\t'+name+'.p4 = ('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*('+name+'.parent2->p4+'+name+'.parent2->p5);\n'
					ans += '\t'+name+'.p5 = (0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*('+name+'.parent2->p4+'+name+'.parent2->p5);\n'
				elif(person.sex == 'female'):
					ans += '\t'+name+'.p1 = ('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p4;\n'
					ans += '\t'+name+'.p2 = (0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p4+('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p5;\n'
					ans += '\t'+name+'.p3 = (0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p5;\n'
					ans += '\t'+name+'.p4 = 0.;\n'
					ans += '\t'+name+'.p5 = 0.;\n'
				elif(person.sex == 'unknown'):
					ans += '\t'+name+'.p1 = 0.5*('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p4;\n'
					ans += '\t'+name+'.p2 = 0.5*((0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p4+('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p5);\n'
					ans += '\t'+name+'.p3 = 0.5*(0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p5;\n'
					ans += '\t'+name+'.p4 = 0.5*('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*('+name+'.parent2->p4+'+name+'.parent2->p5);\n'
					ans += '\t'+name+'.p5 = 0.5*(0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*('+name+'.parent2->p4+'+name+'.parent2->p5);\n'
				else:
					assert 0,'This sex is invalid'
			elif(person.parents[0].sex == 'male' and person.parents[1].sex == 'female'):
				if(person.sex == 'male'):
					ans += '\t'+name+'.p1 = 0.;\n'
					ans += '\t'+name+'.p2 = 0.;\n'
					ans += '\t'+name+'.p3 = 0.;\n'
					ans += '\t'+name+'.p4 = ('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*('+name+'.parent1->p4+'+name+'.parent1->p5);\n'
					ans += '\t'+name+'.p5 = (0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*('+name+'.parent1->p4+'+name+'.parent1->p5);\n'
				elif(person.sex == 'female'):
					ans += '\t'+name+'.p1 = ('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p4;\n'
					ans += '\t'+name+'.p2 = (0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p4+('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p5;\n'
					ans += '\t'+name+'.p3 = (0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p5;\n'
					ans += '\t'+name+'.p4 = 0.;\n'
					ans += '\t'+name+'.p5 = 0.;\n'
				elif(person.sex == 'unknown'):
					ans += '\t'+name+'.p1 = 0.5*('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p4;\n'
					ans += '\t'+name+'.p2 = 0.5*((0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p4+('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p5);\n'
					ans += '\t'+name+'.p3 = 0.5*(0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p5;\n'
					ans += '\t'+name+'.p4 = 0.5*('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*('+name+'.parent1->p4+'+name+'.parent1->p5);\n'
					ans += '\t'+name+'.p5 = 0.5*(0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*('+name+'.parent1->p4+'+name+'.parent1->p5);\n'
				else:
					assert 0,'This sex is invalid'
			elif(person.parents[0].sex == 'unknown' and person.parents[1].sex == 'unknown'):
				if(person.sex == 'male'):
					ans += '\t'+name+'.p1 = 0.;\n'
					ans += '\t'+name+'.p2 = 0.;\n'
					ans += '\t'+name+'.p3 = 0.;\n'
					ans += '\t'+name+'.p4 = 0.5*('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*('+name+'.parent2->p4+'+name+'.parent2->p5)+0.5*('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*('+name+'.parent1->p4+'+name+'.parent1->p5);\n'
					ans += '\t'+name+'.p5 = 0.5*(0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*('+name+'.parent2->p4+'+name+'.parent2->p5)+0.5*(0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*('+name+'.parent1->p4+'+name+'.parent1->p5);\n'
				elif(person.sex == 'female'):
					ans += '\t'+name+'.p1 = 0.5*('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p4+0.5*('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p4;\n'
					ans += '\t'+name+'.p2 = 0.5*((0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p4+('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p5)+0.5*((0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p4+('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p5);\n'
					ans += '\t'+name+'.p3 = 0.5*(0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p5+0.5*(0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p5;\n'
					ans += '\t'+name+'.p4 = 0.;\n'
					ans += '\t'+name+'.p5 = 0.;\n'
				elif(person.sex == 'unknown'):
					ans += '\t'+name+'.p1 = 0.5*(0.5*('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p4+0.5*('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p4);\n'
					ans += '\t'+name+'.p2 = 0.5*(0.5*((0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p4+('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*'+name+'.parent2->p5)+0.5*((0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p4+('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*'+name+'.parent1->p5));\n'
					ans += '\t'+name+'.p3 = 0.5*(0.5*(0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*'+name+'.parent2->p5+0.5*(0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*'+name+'.parent1->p5);\n'
					ans += '\t'+name+'.p4 = 0.5*(0.5*('+name+'.parent1->p1+0.5*'+name+'.parent1->p2)*('+name+'.parent2->p4+'+name+'.parent2->p5)+0.5*('+name+'.parent2->p1+0.5*'+name+'.parent2->p2)*('+name+'.parent1->p4+'+name+'.parent1->p5));\n'
					ans += '\t'+name+'.p5 = 0.5*(0.5*(0.5*'+name+'.parent1->p2+'+name+'.parent1->p3)*('+name+'.parent2->p4+'+name+'.parent2->p5)+0.5*(0.5*'+name+'.parent2->p2+'+name+'.parent2->p3)*('+name+'.parent1->p4+'+name+'.parent1->p5));\n'
				else:
					assert 0,'This sex is invalid'
			else:
				assert 0, 'This combination of parent'
		else:
			assert 0,'This inheritance pattern isn\'t implemented yet!'

	return ans+'\n'

def calcLoss(p,IP):
	name = 'person_'+(str(p.Id).replace('-','_'))
	ans = ''
	if(IP == 'AD'):
		if(p.affected):
			# ans += '\tif('+name+'.p1+'+name+'.p2 == 0) {exit(0);}\n'
			ans += '\tans += log('+name+'.p1+'+name+'.p2);\n'
		else:
			# ans += '\tif('+name+'.p3 == 0) {exit(0);}\n'
			ans += '\tans += log('+name+'.p3);\n'
	elif(IP == 'AR'):
		if(not p.affected):
			# ans += '\tif('+name+'.p1+'+name+'.p2 == 0) {exit(0);}\n'
			ans += '\tans += log('+name+'.p1+'+name+'.p2);\n'
		else:
			# ans += '\tif('+name+'.p3 == 0) {exit(0);}\n'
			ans += '\tans += log('+name+'.p3);\n'
	elif(IP == 'XL'):
		if(p.affected):
			if(p.sex == 'male'):
				# ans += '\tif('+name+'.p5 == 0) {exit(0);}\n'
				ans += '\tans += log('+name+'.p5);\n'
			elif(p.sex == 'female'):
				# ans += '\tif('+name+'.p3 == 0) {exit(0);}\n'
				ans += '\tans += log('+name+'.p3);\n'
			elif(p.sex == 'unknown'):
				# ans += '\tif('+name+'.p3+'+name+'.p5 == 0) {exit(0);}\n'
				ans += '\tans += log('+name+'.p3+'+name+'.p5);\n'
			else:
				assert 0,'Invalid sex'
		else:
			if(p.sex == 'male'):
				# ans += '\tif('+name+'.p4 == 0) {exit(0);}\n'
				ans += '\tans += log('+name+'.p4);\n'
			elif(p.sex == 'female'):
				# ans += '\tif('+name+'.p1+'+name+'.p2 == 0) {exit(0);}\n'
				ans += '\tans += log('+name+'.p1+'+name+'.p2);\n'
			elif(p.sex == 'unknown'):
				# ans += '\tif('+name+'.p1+'+name+'.p2+'+name+'.p4 == 0) {exit(0);}\n'
				ans += '\tans += log('+name+'.p1+'+name+'.p2+'+name+'.p4);\n'
			else:
				assert 0,'Invalid sex'
	else:
		assert 0,'This isn\'t implemented yet'
	return ans+'\n'

def writeLossFunction(people,pedigree,IP):
	# need to write in how to calculate each of their probs
	# finally will calculate the total loss
	ans = 'double ans = 0;\n\n'+personStruct(IP)
	
	startIndex = {}
	last = None
	for i,p in enumerate(pedigree.roots):
		if(IP == 'AD' or IP == 'AR'):
			startIndex[p] = i*2
		elif(IP == 'XL'):
			if(last):
				if(last.sex == 'male'):
					startIndex[p] = startIndex[last]+1
				elif(last.sex == 'female'):
					startIndex[p] = startIndex[last]+2
				elif(last.sex == 'unknown'):
					startIndex[p] = startIndex[last]+4
				else:
					assert 0,'Invalid sex'
			else:
				startIndex[p] = 0
			last = p
		else:
			assert 0,'Not implemented yet'

	for p in people:
		ans += initPerson(p,startIndex,IP)
		ans += calcLoss(p,IP)

	ans += '\treturn -ans;\n'
	return ans

# DO NOT CHANGE THE SPACING OR TABS IN THE TEMPLATES!!!!!!!!
# THIS WILL MAKE ASA NOT RUN!!!!!
def asa_opt_template(IP,pedigree):

	template = '#if FALSE /* ASA_SAVE */\n\
Limit_Acceptances[10000][ASA_TEST:1000]			1000\n\
Limit_Generated[99999]					99999\n\
Limit_Invalid_Generated_States[1000]			1000\n\
Accepted_To_Generated_Ratio[1.0E-6][ASA_TEST:1.0E-4]	1.0E-4\n\
\n\
Cost_Precision[1.0E-18]					1.0E-18\n\
Maximum_Cost_Repeat[5]					5\n\
Number_Cost_Samples[5]					5\n\
Temperature_Ratio_Scale[1.0E-5]				1.0E-5\n\
Cost_Parameter_Scale_Ratio[1.0]				1.0\n\
Temperature_Anneal_Scale[100.0]				100.0\n\
\n\
Include_Integer_Parameters[FALSE=0]			0\n\
User_Initial_Parameters[FALSE=0]			0\n\
Sequential_Parameters[-1]				-1\n\
Initial_Parameter_Temperature[1.0]			1.0\n\
\n\
Acceptance_Frequency_Modulus[100]			100\n\
Generated_Frequency_Modulus[10000]			10000\n\
Reanneal_Cost[1]					1\n\
Reanneal_Parameters[TRUE=1]				1\n\
\n\
Delta_X[0.001]						0.00001\n\
User_Tangents[FALSE=0]					0\n\
Curvature_0[FALSE=0]					0\n\
\n\
___Define_below_if_OPTIONS_FILE_DATA=TRUE\n\
\n\
number_parameters=*parameter_dimension			OTHERADDINGPOINT\n\
\n\
Param#:Minimum:Maximum:InitialValue:Integer[1or2]orReal[-1or-2]\n\
TOREPLACE\n\
___Define_below_if_QUENCH_COST_and_OPTIONS_FILE_DATA=TRUE\n\
\n\
User_Quench_Cost_Scale[0]=1.0				1.0\n\
\n\
___Define_below_if_QUENCH_PARAMETERS_and_QUENCH_COST_and_OPTIONS_FILE_DATA=TRUE\n\
\n\
Param#:User_Quench_Param_Scale[.]=1.0\n\
0	1.0\n\
1	1.0\n\
2	1.0\n\
3	1.0\n\
\n\
NOTE:  Keep all comment lines above, with no extra in-line "white" spaces.\n\
\n\
The OPTIONS_FILE_DATA lines used by ASA_TEST are saved for reference:\n\
\n\
number_parameters=*parameter_dimension			OTHERADDINGPOINT\n\
\n\
Param#:Minimum:Maximum:InitialValue:Integer[1or2]orReal[-1or-2]\n\
0	-10000	10000	999.0			-1\n\
1	-10000	10000	-1007.0			-1\n\
2	-10000	10000	1001.0			-1\n\
3	-10000	10000	-903.0			-1\n\
\n\
/***********************************************************************\n\
* Adaptive Simulated Annealing (ASA)\n\
* Lester Ingber <ingber@ingber.com>\n\
* Copyright (c) 1987-2016 Lester Ingber.  All Rights Reserved.\n\
* ASA-LICENSE file has the license that must be included with ASA code.\n\
***********************************************************************/\n\
\n\
When using this file with ASA_SAVE=TRUE, C code can be added after\n\
the last #endif statement after the  line below, to be recompiled\n\
into the code after the asa_save file is read in.  Be sure to force a\n\
recompile of asa.o and asa_run before restarting runs.  Also be sure you\n\
write the names of these variables as they are used in the asa.c file,\n\
which can differ from their counterparts in asa_usr.c file.  For example,\n\
you might add:\n\
\n\
parameter_maximum[2] = 500;\n\
OPTIONS->Limit_Generated = 700;\n\
OPTIONS->User_Quench_Param_Scale[3] = 0.8; /* assumes QUENCH_PARAMETERS=TRUE */\n\
\n\
$Id: asa_opt,v 30.21 2016/02/02 15:49:45 ingber Exp ingber $\n\
#endif /* ASA_SAVE */\n\
\n\
'
	# this is what the replacement part has to look like
	# 0	0	6.29	0.1			-1\
	# 1	0	3.18	0.1			-1\
	# 2	0	6.29	0.1			-1\
	# 3	0	3.18	0.1			-1\
	# 4	0	6.29	0.1			-1\
	# 5	0	3.18	0.1			-1\
	replacement = ''
	PI = 3.1415926536
	index = 0
	for i,p in enumerate(pedigree.roots):
		if(IP == 'AD' or IP == 'AR'):
			line1 = str(2*i)+'	0	'+str(2*PI)+'	0.1			-1\n'
			line2 = str(2*i+1)+'	-'+str(PI/2)+'	'+str(PI/2)+'	0.1			-1\n'
			replacement += (line1 + line2)
		elif(IP == 'XL'):
			if(p.sex == 'male'):
				line1 = str(index)+'	0	'+str(2*PI)+'	0.1			-1\n'
				index += 1
				replacement += line1
			elif(p.sex == 'female'):
				line1 = str(index)+'	0	'+str(2*PI)+'	0.1			-1\n'
				line2 = str(index+1)+'	-'+str(PI/2)+'	'+str(PI/2)+'	0.1			-1\n'
				index += 2
				replacement += (line1 + line2)
			elif(p.sex == 'unknown'):
				line1 = str(index)+'	0	'+str(2*PI)+'	0.1			-1\n'
				line2 = str(index+1)+'	0	'+str(2*PI)+'	0.1			-1\n'
				line3 = str(index+2)+'	0	'+str(2*PI)+'	0.1			-1\n'
				line4 = str(index+3)+'	0	'+str(2*PI)+'	0.1			-1\n'
				index += 4
				replacement += (line1 + line2 + line3 + line4)
			else:
				assert 0,'Invalid sex'
		else:
			assert 0,'Other inheritance pattern not implemented yet'

	template = template.replace('TOREPLACE',replacement)

	if(IP == 'AD' or IP == 'AR'):
		template = template.replace('OTHERADDINGPOINT',str(len(pedigree.roots)*2))
	elif(IP == 'XL'):
		numbMales = len([x for x in pedigree.roots if x.sex == 'male'])
		numbFemales = len([x for x in pedigree.roots if x.sex == 'female'])
		numbUnknown = len([x for x in pedigree.roots if x.sex == 'unknown'])
		template = template.replace('OTHERADDINGPOINT',str(numbMales*1+numbFemales*2+numbUnknown*4))
	else:
		assert 0,'Not implemented yet'

	return template

def asa_usr_cst_template(equation):

	template = '/***********************************************************************\n\
* Adaptive Simulated Annealing (ASA)\n\
* Lester Ingber <ingber@ingber.com>\n\
* Copyright (c) 1987-2016 Lester Ingber.  All Rights Reserved.\n\
* ASA-LICENSE file has the license that must be included with ASA code.\n\
***********************************************************************/\n\
\n\
 /* $Id: asa_usr_cst.c,v 30.21 2016/02/02 15:49:43 ingber Exp ingber $ */\n\
\n\
 /* asa_usr_cst.c for Adaptive Simulated Annealing */\n\
\n\
#include "asa_usr.h"\n\
\n\
#if COST_FILE\n\
\n\
 /* Note that this is a trimmed version of the ASA_TEST problem.\n\
    A version of this cost_function with more documentation and hooks for\n\
    various templates is in asa_usr.c. */\n\
\n\
 /* If you use this file to define your cost_function (the default),\n\
    insert the body of your cost function just above the line\n\
    "#if ASA_TEST" below.  (The default of ASA_TEST is FALSE.)\n\
\n\
    If you read in information via the asa_opt file (the default),\n\
    define *parameter_dimension and\n\
    parameter_lower_bound[.], parameter_upper_bound[.], parameter_int_real[.]\n\
    for each parameter at the bottom of asa_opt.\n\
\n\
    The minimum you need to do here is to use\n\
    x[0], ..., x[*parameter_dimension-1]\n\
    for your parameters and to return the value of your cost function.  */\n\
\n\
#if HAVE_ANSI\n\
double\n\
cost_function (double *x,\n\
               double *parameter_lower_bound,\n\
               double *parameter_upper_bound,\n\
               double *cost_tangents,\n\
               double *cost_curvature,\n\
               ALLOC_INT * parameter_dimension,\n\
               int *parameter_int_real,\n\
               int *cost_flag, int *exit_code, USER_DEFINES * USER_OPTIONS)\n\
#else\n\
double\n\
cost_function (x,\n\
               parameter_lower_bound,\n\
               parameter_upper_bound,\n\
               cost_tangents,\n\
               cost_curvature,\n\
               parameter_dimension,\n\
               parameter_int_real, cost_flag, exit_code, USER_OPTIONS)\n\
     double *x;\n\
     double *parameter_lower_bound;\n\
     double *parameter_upper_bound;\n\
     double *cost_tangents;\n\
     double *cost_curvature;\n\
     ALLOC_INT *parameter_dimension;\n\
     int *parameter_int_real;\n\
     int *cost_flag;\n\
     int *exit_code;\n\
     USER_DEFINES *USER_OPTIONS;\n\
#endif\n\
{\n\
\n\
  TOREPLACE\n\
\n\
  /* *** Insert the body of your cost function here, or warnings\n\
   * may occur if COST_FILE = TRUE & ASA_TEST != TRUE ***\n\
   * Include ADAPTIVE_OPTIONS below if required */\n\
#if ASA_TEST\n\
#else\n\
#if ADAPTIVE_OPTIONS\n\
  adaptive_options (USER_OPTIONS);\n\
#endif\n\
#endif\n\
\n\
#if ASA_TEST\n\
  double q_n, d_i, s_i, t_i, z_i, c_r;\n\
  int k_i;\n\
  ALLOC_INT i, j;\n\
  static LONG_INT funevals = 0;\n\
\n\
#if ADAPTIVE_OPTIONS\n\
  adaptive_options (USER_OPTIONS);\n\
#endif\n\
\n\
  s_i = 0.2;\n\
  t_i = 0.05;\n\
  c_r = 0.15;\n\
\n\
  q_n = 0.0;\n\
  for (i = 0; i < *parameter_dimension; ++i) {\n\
    j = i % 4;\n\
    switch (j) {\n\
    case 0:\n\
      d_i = 1.0;\n\
      break;\n\
    case 1:\n\
      d_i = 1000.0;\n\
      break;\n\
    case 2:\n\
      d_i = 10.0;\n\
      break;\n\
    default:\n\
      d_i = 100.0;\n\
    }\n\
    if (x[i] > 0.0) {\n\
      k_i = (int) (x[i] / s_i + 0.5);\n\
    } else if (x[i] < 0.0) {\n\
      k_i = (int) (x[i] / s_i - 0.5);\n\
    } else {\n\
      k_i = 0;\n\
    }\n\
\n\
    if (fabs (k_i * s_i - x[i]) < t_i) {\n\
      if (k_i < 0) {\n\
        z_i = k_i * s_i + t_i;\n\
      } else if (k_i > 0) {\n\
        z_i = k_i * s_i - t_i;\n\
      } else {\n\
        z_i = 0.0;\n\
      }\n\
      q_n += c_r * d_i * z_i * z_i;\n\
    } else {\n\
      q_n += d_i * x[i] * x[i];\n\
    }\n\
  }\n\
  funevals = funevals + 1;\n\
\n\
  *cost_flag = TRUE;\n\
\n\
#if FALSE                       /* may set to TRUE if printf() is active */\n\
#if TIME_CALC\n\
  if ((PRINT_FREQUENCY > 0) && ((funevals % PRINT_FREQUENCY) == 0)) {\n\
    printf ("funevals = %ld  ", funevals);\n\
    print_time ("", stdout);\n\
  }\n\
#endif\n\
#endif\n\
\n\
#if ASA_FUZZY\n\
  if (*cost_flag == TRUE\n\
      && (USER_OPTIONS->Locate_Cost == 2 || USER_OPTIONS->Locate_Cost == 3\n\
          || USER_OPTIONS->Locate_Cost == 4)) {\n\
    FuzzyControl (USER_OPTIONS, x, q_n, *parameter_dimension);\n\
  }\n\
#endif /* ASA_FUZZY */\n\
\n\
  return (q_n);\n\
#endif /* ASA_TEST */\n\
}\n\
#endif /* COST_FILE */\n\
'

	template = template.replace('TOREPLACE',equation)
	return template

def asaResult(people,pedigree,inheritancePattern):

	theThing = writeLossFunction(people,pedigree,inheritancePattern)

	asaFolder = './ASA/'

	# fileName = './cEquations/equationC_'+str(inheritancePattern)+'_'+str(pedigree.studyID)+'.txt'
	# with open(fileName,'r') as data_file:
	# 	equation = data_file.read()

	# need to generate a new asa_opt file
	newOptText = asa_opt_template(inheritancePattern,pedigree)
	newOptFile = open(asaFolder+'asa_opt','w+')
	newOptFile.write(newOptText)
	newOptFile.close()

	# need to generate a new asa_usr_cst.c file
	# newCostText = asa_usr_cst_template(equation)
	newCostText = asa_usr_cst_template(theThing)
	newCostFile = open(asaFolder+'asa_usr_cst.c','w+')
	newCostFile.write(newCostText)
	newCostFile.close()


	# need to compile and run
	currentDir = os.getcwd()
	os.chdir(asaFolder)
	os.system('g++ -g -DOPTIONS_FILE_DATA=TRUE -o asa_run asa_usr.c asa_usr_cst.c asa.c')
	time.sleep(3)
	os.system('./asa_run')

	with open('asa_out','r') as data_file:
		asa_output = data_file.read()

	asaOutputFile = open('../pedigreeOutputs/asa_out_'+str(inheritancePattern)+'_'+str(pedigree.studyID)+'.txt','w+')
	asaOutputFile.write(asa_output)
	asaOutputFile.close()

	# need to extract results
	with open('asa_usr_out','r') as data_file:
		everything = data_file.read()
		lines = everything.split('\n')	

	asaOutputFile2 = open('../pedigreeOutputs/asa_usr_out_'+str(inheritancePattern)+'_'+str(pedigree.studyID)+'.txt','w+')
	asaOutputFile2.write(everything)
	asaOutputFile2.close()

	ansList = lines[3]
	paramLines = lines[5:-1]

	ans = np.float128(ansList[19:].strip(' ').strip('\n'))
	params = [np.float128(l.split('\t')[-1].strip(' ')) for l in paramLines]
	print('Got: '+str(ans)+' (Real: '+str(pedigree.inheritancePattern)+')')

	os.chdir(currentDir)

	return [ans,params]
	



