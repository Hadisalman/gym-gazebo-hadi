import numpy as np
from XYrot import XYrot

def stabilizationPID(cpg):

    #Gives a rotation matrix for correct orientation given the world frame
    R = XYrot(cpg['pose'])[0]

    FKbody = cpg['requestedLegPositions']
    FKworld = np.dot(cpg['pose'], FKbody.T)

    FKworldCor = np.dot(R[:3,:3] , FKworld)
    cpg['feetTemp'] = FKworldCor

    # Set the stance to the lowest tripod
    use145 = np.mean(FKworldCor[2,[0,3,4]]) < np.mean(FKworldCor[2,[1,2,5]])
    cpg['isStance'][0,[0,3,4]] = use145
    cpg['isStance'][0,[1,2,5]] = use145 ^ 1

    ## Compute Height from Ground Plane

    dirVec = [[0],[0],[1]]

    # Get positions of feet on ground
    #groundPositions = FKworldCor[:,list(cpg['isStance'][0].astype(bool))].reshape(3,6)  # wrong Bill
    #cpg['planeTemp']= FKworldCor  # wrong Bill
    #cpg['planePoint']= groundPositions[:,0]  # wrong Bill


    #Determine normal of plane formed by ground feet
    if use145:
        cpg['groundNorm']= np.cross((FKworldCor[:,0] - FKworldCor[:,3]).T, (FKworldCor[:,0] - FKworldCor[:,4]).T)
    else:
        cpg['groundNorm']= np.cross((FKworldCor[:,1] - FKworldCor[:,2]).T, (FKworldCor[:,1] - FKworldCor[:,5]).T)

    cpg['groundD']= np.dot(cpg['groundNorm'], FKworldCor[:,0])

    #find intersection
    #t = np.linalg.lstsq((np.dot(cpg['groundNorm'] , dirVec)).T,(cpg['groundD']).T)[0].T  # wrong Bill
    t = cpg['groundD'] / np.dot(cpg['groundNorm'] , dirVec)

    #find height
    cpg['zDist']= np.linalg.norm(dirVec * t)


     ## Adjust legs for Z

    cpg['zHistory'][0,cpg['zHistoryCnt']] = cpg['zDist']
    cpg['zHistoryCnt']= cpg['zHistoryCnt']+ 1
    cpg['zHistoryCnt']= np.mod(cpg['zHistoryCnt'],9) + 1

    zErr = cpg['bodyHeight']- np.median(cpg['zHistory'])

    FKworldCor[2] = FKworldCor[2] + zErr


    cpg['dxWorld']= FKworldCor - FKworld
    print('dxWorld')
    print(cpg['dxWorld'])

    cpg['dx']= - np.linalg.lstsq(cpg['pose'], cpg['dxWorld'])[0]
    print('dx')
    print(cpg['dx'])

    return cpg
