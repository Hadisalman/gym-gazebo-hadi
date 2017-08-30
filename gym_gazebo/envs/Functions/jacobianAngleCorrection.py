import numpy as np
def jacobianAngleCorrection(cpg):
    ## Calculate Angle Correction

    J = cpg['smk'].getLegJacobians(cpg['legs'])
    print('J')
    print(J[0:3,:,1])
    print('------')

    theta = np.zeros((3,6))
    numLegs = 6

    for leg in range(numLegs):
        theta[:,leg] = np.linalg.lstsq(J[:3,:,leg], cpg['dx'][:,leg])[0][0]

    return theta
