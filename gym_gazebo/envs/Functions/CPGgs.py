import numpy as np
from math import atan, sqrt, pi
from stabilizationPID import stabilizationPID
from jacobianAngleCorrection import jacobianAngleCorrection


def CPGgs(cpg, t, dt):

    theta = np.zeros((3,6))

    # if t > cpg['initLength']:
    #     cpg = stabilizationPID(cpg)
    #     theta = jacobianAngleCorrection(cpg)
        

    # else:
    cpg['legs'] = np.zeros((1,18))


#Filter large spikes in theta
    theta = np.array([[angle * (abs(angle) <= 1.0) for angle in row] for row in theta])

#Move back to uncorrected stance
    alpha = 0.1
    #beta = 0.1

    theta[1] = theta[1] - alpha * cpg['theta2']
    #theta[2] = np.dot(theta[2] - beta , cpg['theta3'])


    ## Applies CPG to the first two joints and IK to the last joint. Only apply stabilization to second joint though

    # height
    a = 0.6 * np.ones((1, 6))#[.3 .3 .3 .3 .3 .3]; # semi major-axis of the limit-ellipse  #tilt
    b = pi/18 *np.ones((1, 6)) # semi minor-axis of the limit-ellipse

    #step height
    shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
    shoulders1Corr      = np.array([1,-1,1,-1,1,-1]) * cpg['direction'] # correction factor for left/right legs
    shoulder1Offsets    = np.array([-1,-1,0,0,1,1]) * pi/3 * cpg['direction'] # offset so that legs are more spread out
    shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
    shoulder2Offsets    = pi/5 * np.ones((1,6)) #tilt
    elbows              = list(range(2,18,3)) # joint IDs of the elbow joints

    # Robot Dimensions
    endWidth = 0.075 # dimensions of the end section of the leg
    endHeight = 0.185
    endTheta = atan(endHeight/endWidth)
    L1 = 0.125 # all distances in m
    L2 = sqrt(endWidth**2 + endHeight**2)
    moduleLen = .097

    xKy = 0.08 # distance of the central leg from the shoulder
    GSoffset = 0.07
    gammaX = 20
    gammaY = 20
    sweep = pi/24


    radCentral = L1*np.cos(shoulder2Offsets[0][0]) + .063-.0122
    d = 2 * np.tan(sweep/(2*b)) * (moduleLen + radCentral)
    r0 = moduleLen + np.array([xKy,xKy,radCentral,radCentral,xKy,xKy]) - GSoffset

    K = [[ 0,-1,-1, 1, 1,-1],
         [-1, 0, 1,-1,-1, 1],
         [-1, 1, 0,-1,-1, 1],
         [ 1,-1,-1, 0, 1,-1],
         [ 1,-1,-1, 1, 0,-1],
         [-1, 1, 1,-1,-1, 0]]

# CPG Equations
    dx = (gammaX * (1- (cpg['x'][t]**2)/(b**2) - ((cpg['y'][t] - cpg['theta2'])**2)/(a**2))
          *cpg['x'][t] - cpg['w_y'] * b / a * (cpg['y'][t] - cpg['theta2']))
    dy = (gammaY * (1- (cpg['x'][t]**2)/(b**2) - ((cpg['y'][t] - cpg['theta2'])**2)/(a**2))
        *(cpg['y'][t] - cpg['theta2']) + cpg['w_y'] * a / b * cpg['x'][t] + (np.dot(K,(cpg['y'][t].T) - cpg['theta2']).T).T/4 + theta[1])

    cpg['theta2'] += theta[1] * dt
    #cpg['theta3'] += theta[2] * dt

    if not cpg['move']:
        dx = 0
        dy = theta[1]

    cpg['x'][t+1] = cpg['x'][t] + dx * dt
    cpg['y'][t+1] = cpg['y'][t] + dy * dt

## CPG
    r0s = r0 * xKy / r0[2]
    cpg['legs'][0,shoulders1] = (shoulder1Offsets+cpg['x'][t+1]) * shoulders1Corr #CPG Controlled
    cpg['legs'][0,shoulders2] = (shoulder2Offsets+np.maximum(cpg['theta2'], cpg['y'][t+1]))
    
    #cor = zeros(1,6);
    #if t > cpg.initLength
        #dead = 0.3;
        #cor = 1.5 * (max(cpg.y(t+1,:) - cpg.theta2,dead) - dead)
        
    #%     dead = 0.1;
    #%     cor = 1 * (max(cpg.y(t+1,:),dead) - dead);
    #end

    cpg['legs'][0,elbows] = np.arcsin( (r0s/np.cos(cpg['legs'][0,shoulders1]) - L1*np.cos(cpg['legs'][0,shoulders2]))/L2) - cpg['legs'][0,shoulders2] - pi/2 + endTheta
    
    return cpg
