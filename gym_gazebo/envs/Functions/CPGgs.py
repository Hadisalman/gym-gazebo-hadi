import numpy as np
# from Functions.stabilizationPID import stabilizationPID
# from Functions.jacobianAngleCorrection import jacobianAngleCorrection
#from Functions.get2b import get2b
#from Functions.getDistance2 import getDistance2


# CPGgas2.py

def CPGgs(cpg, t, dt):

    theta = np.zeros((3,6))

    #if t > cpg['initLength']:
    #    cpg = stabilizationPID(cpg)
    #    theta = jacobianAngleCorrection(cpg)
    #else:
    cpg['legs'] = np.zeros((1,18))


    #Filter large spikes in theta
    theta = np.array([[angle * (abs(angle) <= 1.0) for angle in row] for row in theta])

    #Move back to uncorrected stance
    #theta in rad
    alpha = 0.1
    theta = np.rad2deg(theta)
    theta[1] -= alpha * cpg['theta2']
    
    ## Applies CPG to the first two joints and IK to the last joint. Only apply stabilization to second joint though

    shoulders1          = list(range(0,18,3)) # joint IDs of the shoulders
    shoulders2          = list(range(1,18,3)) # joint IDs of the second shoulder joints
    elbows              = list(range(2,18,3)) # joint IDs of the elbow joints

    shoulders1Corr      = np.array([1,-1,1,-1,1,-1]) * cpg['direction'] # correction factor for left/right legs
    shoulder1Offsets    = np.array([-1,-1,0,0,1,1]) * cpg['s1Off'] * cpg['direction'] # offset so that legs are more spread out
    # shoulder2Offsets    = (np.pi/16) * np.ones((1,6)) #tilt
    shoulder2Offsets    = cpg['s2Off']* np.ones((1,6)) #tilt

    # Robot Dimensions
    endWidth = 0.075 # dimensions of the end section of the leg
    endHeight = 0.185
    endTheta = np.arctan(endHeight/endWidth)
    L1 = 0.125 # all distances in m
    L2 = np.sqrt(endWidth**2 + endHeight**2)
    moduleLen = .097

    # xKy = 0.1 # distance of the central leg from the shoulder
    xKy = cpg['r']
    GSoffset = 0.07

    gammaX = 20
    gammaY = 20
    sweep = np.pi/24 

    radCentral = L1*np.cos(shoulder2Offsets[0,0]) + .063-.0122 # a value
    d = 2 * np.tan(sweep / (2*cpg['b'])) * (moduleLen + radCentral) #2D array
    r0 = moduleLen + np.array([xKy,xKy,radCentral,radCentral,xKy,xKy]) - GSoffset # 1D array
    mu = np.arccos(np.sqrt((4*r0**2 + np.sqrt(2)*np.sqrt(r0**2 * (d**2 + 8*r0**2 - d**2*np.cos(4*shoulder1Offsets))) + 2* d**2 *np.sin(shoulder1Offsets)**2)/(d**2 + 4*r0**2))/np.sqrt(2)) # 2D array
    r0s = r0 * xKy / r0[2]

    ###################################

    alpha = np.absolute(shoulder1Offsets)
    b = np.deg2rad(cpg['b'])
    cpg['l0'] = r0s/np.cos(alpha - b)
    cpg['l1'] = r0s/np.cos(alpha)
    cpg['l2'] = r0s/np.cos(alpha + b)
    cpg['d1'] = np.sqrt(cpg['l1']**2 + cpg['l0']**2 - 2*cpg['l1']*cpg['l0']*np.cos(b))
    cpg['d2'] = np.sqrt(cpg['l1']**2 + cpg['l2']**2 - 2*cpg['l1']*cpg['l2']*np.cos(b))

    tmp1 = cpg['d1'][0][4]
    tmp2 = cpg['d1'][0][5]
    cpg['d1'][0][4:6:1] = cpg['d2'][0][4:6:1]
    cpg['d2'][0][4] = tmp1
    cpg['d2'][0][5] = tmp2
    cpg['phi0'] = np.pi/2 - np.array([1,1,0,0,1,1]) * cpg['s1Off']


    ###################################

    # steplen = np.absolute(2 * r0s[2] * np.tan(cpg['b0'][0][2])) # cpg['b'] is based on degree or radian?

    (cpg, bp, phi1) = getb2(cpg, shoulder1Offsets);

    K = [[ 0,-1,-1, 1, 1,-1],
         [-1, 0, 1,-1,-1, 1],
         [-1, 1, 0,-1,-1, 1],
         [ 1,-1,-1, 0, 1,-1],
         [ 1,-1,-1, 1, 0,-1],
         [-1, 1, 1,-1,-1, 0]]

    # CPG Equations
    n = 4
    gamma = 1.0
    lambd = 6.0

    x0 = np.zeros((1,6))
    y0 = cpg['theta2']
    #print(y0)

    ## Normal ellipse
    #dx = -2 * cpg['a'] * (cpg['y'][t] - y0)  * cpg['w'] + gamma * (cpg['mu']**2 - (cpg['b'] * cpg['x'][t]**2 + cpg['a'] * (cpg['y'][t] - y0)**2)) * 2 * cpg['b'] *  cpg['x'][t]
    #dy =  2 * cpg['b'] *  cpg['x'][t]        * cpg['w'] + gamma * (cpg['mu']**2 - (cpg['b'] * cpg['x'][t]**2 + cpg['a'] * (cpg['y'][t] - y0)**2)) * 2 * cpg['a'] * (cpg['y'][t] - y0) + (np.dot(K,(cpg['y'][t] - y0).T)).T/lambd + theta[1]
    
    H = (np.absolute((cpg['x'][t] - x0) / cpg['b']) ** n + np.absolute((cpg['y'][t] - y0) / cpg['a']) ** n)
    dHdx = (n * (cpg['x'][t] - x0) * (np.absolute((cpg['x'][t] - x0)/cpg['b']) ** (n-2))) / (cpg['b'] ** 2)
    dHdy = (n * (cpg['y'][t] - y0) * (np.absolute((cpg['y'][t] - y0)/cpg['a']) ** (n-2))) / (cpg['a'] ** 2)
    dx = dHdx * gamma * (1-H) - cpg['w'] * dHdy
    dy = dHdy * gamma * (1-H) + cpg['w'] * dHdx + theta[1] + np.dot(K, (cpg['y'][t] - cpg['theta2']).T).T / lambd + theta[1]

    # H = (abs((cpg.x(t,:) - x0)./(cpg.b)).^n + abs((cpg.y(t,:) - y0)./(cpg.a)).^n);
    # dHdx = (n .* (cpg.x(t,:) - x0) .* (abs((cpg.x(t,:) - x0)./cpg.b).^(n-2)))./(cpg.b.^2);
    # dHdy = (n .* (cpg.y(t,:) - y0) .* (abs((cpg.y(t,:) - y0)./cpg.a).^(n-2)))./(cpg.a.^2);

    # dx = dHdx .* gamma .* (1-H) - cpg.w .* dHdy;
    # dy = dHdy .* gamma .* (1-H) + cpg.w .* dHdx + (K * (cpg.y(t,:) - cpg.theta2)')'./lambda + theta(2,:);

    cpg['theta2'] += theta[1] * dt

    if not cpg['move']:
        dx = 0
        dy = theta[1]
    # print(dx)
    # print(dy)
    cpg['x'][t+1,:] = cpg['x'][t,:] + dx * dt
    cpg['y'][t+1,:] = cpg['y'][t,:] + dy * dt

    ## CPG
    r0s = r0 * xKy / r0[2]
    xout = cpg['x'][t+1] * np.rad2deg(bp) / (2.0 * cpg['b'])
    t2rad = np.deg2rad(cpg['theta2'])

    #cpg['legs'][0,shoulders1] = (shoulder1Offsets + cpg['x'][t+1]) * shoulders1Corr #CPG Controlled
    cpg['legs'][0,shoulders1] = (shoulder1Offsets + np.deg2rad(xout)) * shoulders1Corr #CPG Controlled
    cpg['legs'][0,shoulders2] = (shoulder2Offsets + np.maximum(t2rad, np.deg2rad(cpg['y'][t+1])))


    cor = np.zeros((1,6))
    if t > cpg['initLength']:
        dead = 0.3
        cor = cpg['t3Str'] * (np.maximum(np.deg2rad(cpg['y'][t+1]) - t2rad, dead) - dead)
    
    #sinVal = np.maximum(-np.ones((1,6)), np.minimum(np.ones((1,6)), (r0s/np.cos(cpg['legs'][0,shoulders1]) - L1*np.cos(cpg['legs'][0,shoulders2]))/L2))
    #cpg['legs'][0,elbows] = np.arcsin( sinVal ) - cpg['legs'][0,shoulders2] - np.pi/2 + endTheta


    dist = getDistance2(cpg,phi1,np.deg2rad(xout),shoulder1Offsets,shoulders1Corr)
    #print('npcoscpdlegs0shoulders1')
    #print(np.cos(cpg['legs'][0,shoulders1]))
    #print("l1")
    #print(L1)
    #print("npcoscpglegs0shoulders2")
    #print(np.cos(cpg['legs'][0, shoulders2]))
    #print("l2")
    #print(L2)
    #tempx = (dist / np.cos(cpg['legs'][0,shoulders1]) - L1 * np.cos(cpg['legs'][0,shoulders2])) / L2
    #print("tempx")
    #print(tempx)
    cpg['legs'][0,elbows] = (np.arcsin( (dist / np.cos(cpg['legs'][0,shoulders1]) - L1 * np.cos(cpg['legs'][0,shoulders2])) / L2) - cpg['legs'][0,shoulders2] - np.pi/2 + endTheta) + cor

    return cpg




def getb2(cpg,shoulder1Offset):

    theta = cpg['phi']
    if (theta>=0 and theta<=np.pi):
        sgn = 1 
    else: sgn = 0

    if not sgn:
        theta = theta - np.pi
        sgn = -1

    alpha = np.absolute(shoulder1Offset)
    phi0 = np.pi/2 - alpha
    phi1 = np.ones((1,6))
    limit = np.array([[np.pi - phi0[0][0], phi0[0][1], np.pi/2, np.pi/2, phi0[0][4], np.pi-phi0[0][5]]])
    th1 = np.array([[phi0[0][0]+theta, phi0[0][1]-theta, np.pi/2+theta, np.pi/2-theta, np.pi-phi0[0][4]+theta, np.pi-phi0[0][5]-theta]])
    th2 = np.array([[theta-np.pi+phi0[0][0], np.pi-theta+phi0[0][1], theta-np.pi/2, 3/2*np.pi-theta, theta-phi0[0][4], 2*np.pi-theta-phi0[0][5]]])
    for i in range(6):
        phi1[0][i] = th1[0][i]
        if theta>limit[0][i]:
            phi1[0][i] = th2[0][i]

    phi2 = np.pi - phi1
    l3 = np.sqrt(cpg['l1']**2 + cpg['d1']**2 - 2.0*cpg['l1']*cpg['d1']*np.cos(phi1))
    l4 = np.sqrt(cpg['d2']**2 + cpg['l2']**2 - 2.0*cpg['l2']*cpg['d2']*np.cos(phi2))
    bp = np.absolute(np.arcsin(cpg['d1']*np.sin(phi1)/l3)) * sgn;
    bp2 = np.absolute(np.arcsin(cpg['d2']*np.sin(phi2)/l4)) * sgn;
    sgn = np.ones((1,6))

    if theta>(np.pi/6):
        sgn[0][1] = -1
        sgn[0][4] = -1

    if theta>(np.pi/2):
        sgn[0][2] = -1
        sgn[0][3] = -1

    if theta>(np.deg2rad(150)):
        sgn[0][0] = -1
        sgn[0][5] = -1

    bp = (bp + bp2) * sgn

    return (cpg, bp, phi1)




def getDistance2(cpg, phi1, x, s1offsets, shoulders1Corr):
    #print(len(x))
    #print("cpgl1")
    #print(cpg['l1'])
    #print("phi1")
    #print(phi1)
    #print('x')
    #print(x)
    #print("s1oddsets")
    #print(s1offsets)
    #print("shoulders1Corr")
    #print(shoulders1Corr)
    dist = np.zeros((1,6))

    i = 0;

    for ele in x[0]:
        if ele>0:
            x[0][i]=1
        else: x[0][i]=0
    idx = np.array(x,dtype=bool)

    
    dist[idx] = cpg['l1'][idx] * np.sin(phi1[idx]) / np.sin(np.pi-phi1[idx]-np.absolute(x[idx]))
    ang = np.absolute(s1offsets) + np.absolute(x) * np.array([[-1,-1,1,1,1,1]])
    dist[idx] = dist[idx] * np.cos(ang[idx])

    phi2 = np.pi-phi1
    dist[~idx] = cpg['l1'][~idx] * np.sin(phi2[~idx]) / np.sin(np.pi-phi2[~idx]-np.absolute(x[~idx]))
    ang = np.absolute(s1offsets) + np.absolute(x) * np.array([[1,1,1,1,-1,-1]])
    dist[~idx] = dist[~idx] * np.cos(ang[~idx])
    return dist



# L1, L2, shoulders1, shoulders2 = 0.125, 0.1996, list(range(0,18,3)), list(range(1,18,3))