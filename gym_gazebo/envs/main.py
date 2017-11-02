'''
Snake Monster stabilization with CPG controls
 Created 24 May 2017
 requires the HEBI API and the Snake Monster HEBI API

 Setting up the Snake Monster

 NOTE: If the modules have been changed, first correct the names and run
calibrateSM.m
'''
from IPython import embed
from copy import copy
import time
import numpy as np
import rospy
from std_msgs.msg import Float64
# import hebiapi
#import setupfunctions as setup
import tools
# import SMCF
# from SMCF.SMComplementaryFilter import feedbackStructure,decomposeSO3
# import seatools.hexapod as hp
from Functions.Controller import Controller
from Functions.CPGgs import CPGgs


def publish_commands( hz ):
    pub={}
    ns_str = '/snake_monster'
    cont_str = 'eff_pos_controller'
    for i in xrange(6) :
        for j in xrange(3) :
            leg_str='L' + str(i+1) + '_' + str(j+1)
            pub[leg_str] = rospy.Publisher( ns_str + '/' + leg_str + '_'
                                            + cont_str + '/command',
                                            Float64, queue_size=10 )
    rospy.init_node('walking_controller', anonymous=True)
    rate = rospy.Rate(hz)
    jntcmds = JointCmds()
    while not rospy.is_shutdown():
        jnt_cmd_dict = jntcmds.update(1./hz)
        for jnt in jnt_cmd_dict.keys() :
            pub[jnt].publish( jnt_cmd_dict[jnt] )
        rate.sleep()

rospy.init_node('walking_controller', anonymous=True)
pub={}
ns_str = '/snake_monster'
cont_str = 'eff_pos_controller'
for i in xrange(6) :
    for j in xrange(3) :
        leg_str='L' + str(i+1) + '_' + str(j+1)
        pub[leg_str] = rospy.Publisher( ns_str + '/' + leg_str + '_'
                                            + cont_str + '/command',
                                            Float64, queue_size=10 )

print('Setting up Snake Monster...')

# names = SMCF.NAMES
names = ['SA012', 'SA059', 'SA030',
        'SA058', 'SA057', 'SA001',
        'SA078', 'SA040', 'SA048',
        'SA081', 'SA050', 'SA018',
        'SA046', 'SA032', 'SA026',
        'SA041', 'SA072', 'SA077']

#HebiLookup = tools.HebiLookup
# embed()
# shoulders = names[::3]
# imu = HebiLookup.getGroupFromNames(shoulders)
#snakeMonster = HebiLookup.getGroupFromNames(names)
# while imu.getNumModules() != 6:
#     print('Found {} modules in shoulder group, {} in robot.'.format(imu.getNumModules(), snakeMonster.getNumModules()), end='  \r')
#     imu = HebiLookup.getGroupFromNames(shoulders)
# print('Found {} modules in shoulder group, {} in robot.'.format(imu.getNumModules(), snakeMonster.getNumModules()))
# snakeData = setup.setupSnakeMonsterShoulderData()
# smk = hp.HexapodKinematics()

# fbk = feedbackStructure(imu)
# gyroOffset, accelOffset = SMCF.calibrateOffsets(fbk)
# #setup.setupSnakeMonster()

cmd = tools.CommandStruct()
# CF = SMCF.SMComplementaryFilter(accelOffset=accelOffset, gyroOffset=gyroOffset)
# fbk.getNextFeedback()
# CF.update(fbk)
# time.sleep(0.02)
# pose = []
# while pose is None or not list(pose):
#     fbk.getNextFeedback()
#     CF.update(fbk)
#     pose = copy(CF.R)


print('Setup complete!')

## Initialize Variables

T = 600
dt = 0.02
nIter = round(T/dt)

cpg = {
    'initLength': 250,
    'w_y': 2.0,
    'bodyHeight':0.13,
    'bodyHeightReached':False,
    'zDist':0,
    'zHistory':np.ones((1,10)),
    'zHistoryCnt':0,
    'direction': np.ones((1,6)),
    'x':3 * np.array([[.11, -.1, .1, -.01, .12, -.12]]+[[0, 0, 0, 0, 0, 0] for i in range(30000)]),
    'y':np.zeros((30000+1,6)),
    'forward': np.ones((1,6)),
    'backward': -1 * np.ones((1,6)),
    'leftturn': [1, -1, 1, -1, 1, -1],
    'rightturn': [-1, 1, -1, 1, -1, 1],
    'legs': np.zeros((1,18)),
    'requestedLegPositions': np.zeros((3,6)),
    'correctedLegPositions': np.zeros((3,6)),
    'realLegPositions': np.zeros((3,6)),
    #'smk': smk,
    'isStance':np.zeros((1,6)),
    'pose': np.identity(3),
    'move':True,
    'groundNorm':np.zeros((1,3)),
    'groundD': 0,
    'gravVec':np.zeros((3,1)),
    'planePoint': [[0], [0], [-1.0]],
    'theta2':0,
    'theta3':0,
    'theta2Trap': 0,
    'groundTheta':np.zeros((1,30000)),
    'yOffset':np.zeros((1,6)),
    'eOffset':np.zeros((1,6)),
    'theta3Trap': 0,
    'planeTemp':np.zeros((3,3)),
    'feetTemp':np.zeros((3,6)),
    'yReq':0,
    'o':0,
    'poseLog':[],
    'feetLog':[],
    'planeLog':[]
    }


cpg['zHistory'] = cpg['zHistory'] * cpg['bodyHeight']
## Walk the Snake Monster

print('Finding initial stance...')

# joy = Controller()
# cpgJoy = True

shoulders2 = list(range(2,18,3))
elbows = list(range(3,18,3))

sampleIter = 600
cnt = 0


### BILL TESTING (doesn't apss for now)
#shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
#shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
#elbows              = list(range(2,18,3)) # joint IDs of the elbow joints
#cpg['legs'][0,shoulders1] = [-3.14/4, 3.14/4, 0, 0, 3.14/4, -3.14/4]; #% offset so that legs are more spread out
#cpg['legs'][0,shoulders2] = [3.14/2, 3.14/2, 3.14/2, 3.14/2, 3.14/2, 3.14/2];
#cpg['legs'][0,elbows] = [-3.14/2, -3.14/2, -3.14/2, -3.14/2, -3.14/2, -3.14/2];
#cmd.position = cpg['legs']; #%FJLFJLKJSDLKFJLSKJFLKSJFL
#snakeMonster.setAngles(cmd.position[0]);    
#print(smk.getLegPositions(cpg['legs']))
#J = cpg['smk'].getLegJacobians(cpg['legs'])
#print('J')
#print(J)
#while True:
    #asdf = 0

for t in range(30000):
  # tStart = time.perf_counter()

#    if t == cpg['initLength']:
#        print('Snake Monster is ready!')
#        print('Begin walking')

    ## Joystick stuffs - Exit Button
#    if joy.pressed[2]:
 #       resetSnakeMonster()
  #      print('Reset.\n')
   #     print('Exiting at t = {}\n'.format(t))
    #    break

    #if joy.pressed[1]:
     #   print('Exiting at t = {}\n'.format(t))
      #  break
    
    ##Get pose/gravity vector
    # fbk.getNextFeedback()
    # CF.update(fbk)
    #cpg['pose']= copy(CF.R)
    
    # TEMPORARY print


    #cpg['gravVec']= np.linalg.lstsq(cpg['pose'][:3, :3], [[0],[0],[-1]])[0]
    #cpg['poseLog'].append(cpg['pose'][:3, :3])

    # Get leg positions

    #realLegs = snakeMonster.getAngles()[:18]
    #cpg['realLegPositions']= smk.getLegPositions(realLegs)


    # Apply CPG

    '''
    if t >= cpg['initLength']and cpgJoy:
        if any([c != 0 for c in joy.channel[:2]]):
            cpg['move']= True
            if joy.channel(2) > 0:
                cpg['direction']= cpg['forward']
            elif joy.channel(2) < 0:
                cpg['direction']= cpg['backward']
            elif joy.channel(1) > 0:
                cpg['direction']= cpg['rightturn']
            elif joy.channel(1) < 0:
                cpg['direction']= cpg['leftturn']
        else:
            cpg['move']= false
    '''
    #cpg['requestedLegPositions']= smk.getLegPositions(cpg['legs'])
    
    cpg['direction']= cpg['forward']

    cpg = CPGgs(cpg, t, dt)

    cpg['feetLog'].append(cpg['feetTemp'])
    #cpg['planeLog'].append(cpg['planeTemp'])

    # Command
    cmd.position = cpg['legs']

    
        
    pub['L'+'1'+'_'+'1'].publish(cmd.position[0][0])
    pub['L'+'1'+'_'+'2'].publish(cmd.position[0][1])
    pub['L'+'1'+'_'+'3'].publish(cmd.position[0][2])
    pub['L'+'6'+'_'+'1'].publish(cmd.position[0][3])
    pub['L'+'6'+'_'+'2'].publish(cmd.position[0][4])
    pub['L'+'6'+'_'+'3'].publish(cmd.position[0][5])
    pub['L'+'2'+'_'+'1'].publish(cmd.position[0][6])
    pub['L'+'2'+'_'+'2'].publish(cmd.position[0][7])
    pub['L'+'2'+'_'+'3'].publish(cmd.position[0][8])
    pub['L'+'5'+'_'+'1'].publish(cmd.position[0][9])
    pub['L'+'5'+'_'+'2'].publish(cmd.position[0][10])
    pub['L'+'5'+'_'+'3'].publish(cmd.position[0][11])
    pub['L'+'3'+'_'+'1'].publish(cmd.position[0][12])
    pub['L'+'3'+'_'+'2'].publish(cmd.position[0][13])
    pub['L'+'3'+'_'+'3'].publish(cmd.position[0][14])
    pub['L'+'4'+'_'+'1'].publish(cmd.position[0][15])
    pub['L'+'4'+'_'+'2'].publish(cmd.position[0][16])
    pub['L'+'4'+'_'+'3'].publish(cmd.position[0][17])

    # print(cmd.position)
    
    ###cmd.torque = gravComp(smk, cpg['legs'], cpg['gravVec'])

    #snakeMonster.set(cmd);
# snakeMonster.setAngles(cmd.position[0])
    #snakeMonster.setTorques(cmd.torque)

# loopTime = time.perf_counter() - tStart
# time.sleep(max(0,dt-loopTime))
    #joy.running = False
    # pause('Program completed')
