       �K"	   �ǩ�Abrain.Event:2�`n���      �*	HP$�ǩ�A"��
z
flatten_1_inputPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
^
flatten_1/ShapeShapeflatten_1_input*
_output_shapes
:*
T0*
out_type0
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
Y
flatten_1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
\
flatten_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
m
dense_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�~ٽ*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�~�=*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
_output_shapes
:	�*
seed2���*
seed���)*
T0*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes
:	�

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	�
�
dense_1/kernel
VariableV2*
shape:	�*
shared_name *
dtype0*
_output_shapes
:	�*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	�
|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
z
dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
]
activation_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
m
dense_2/random_uniform/shapeConst*
valueB"   d   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�d*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	�d

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�d*
T0
�
dense_2/kernel
VariableV2*
shape:	�d*
shared_name *
dtype0*
_output_shapes
:	�d*
	container 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�d
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�d
Z
dense_2/ConstConst*
dtype0*
_output_shapes
:d*
valueBd*    
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:d
�
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
\
activation_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:���������d
m
dense_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"d   2   
_
dense_3/random_uniform/minConst*
valueB
 *��L�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes

:d2*
seed2�ځ*
seed���)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:d2
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:d2
�
dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking(
{
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes

:d2*
T0
Z
dense_3/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
x
dense_3/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:2*
	container *
shape:2
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:2
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:2*
T0*
_class
loc:@dense_3/bias
�
dense_3/MatMulMatMulactivation_2/Reludense_3/kernel/read*
T0*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������2
\
activation_3/ReluReludense_3/BiasAdd*'
_output_shapes
:���������2*
T0
m
dense_4/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:2*
seed2��K
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

:2*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
T0*
_output_shapes

:2
�
dense_4/kernel
VariableV2*
dtype0*
_output_shapes

:2*
	container *
shape
:2*
shared_name 
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:2*
use_locking(
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:2
Z
dense_4/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
x
dense_4/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_4/bias
q
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0
�
dense_4/MatMulMatMulactivation_3/Reludense_4/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
d
activation_4/IdentityIdentitydense_4/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_5/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
_output_shapes
: *
valueB
 *�m�*
dtype0
_
dense_5/random_uniform/maxConst*
valueB
 *�m?*
dtype0*
_output_shapes
: 
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2��*
seed���)
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes

:
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
_output_shapes

:*
T0
�
dense_5/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
{
dense_5/kernel/readIdentitydense_5/kernel*!
_class
loc:@dense_5/kernel*
_output_shapes

:*
T0
Z
dense_5/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_5/bias
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
_output_shapes
:*
T0
�
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
m
lambda_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:���������*
Index0*
T0
b
lambda_1/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
lambda_1/ExpandDims
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
o
lambda_1/strided_slice_1/stackConst*
_output_shapes
:*
valueB"       *
dtype0
q
 lambda_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*
T0*'
_output_shapes
:���������
o
lambda_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
_
lambda_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*
T0*
_output_shapes

:*

Tidx0*
	keep_dims(
b
lambda_1/subSublambda_1/addlambda_1/Mean*
T0*'
_output_shapes
:���������
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*"
_class
loc:@Adam/iterations
v
Adam/iterations/readIdentityAdam/iterations*"
_class
loc:@Adam/iterations*
_output_shapes
: *
T0	
Z
Adam/lr/initial_valueConst*
valueB
 *o�9*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: *
use_locking(
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
_output_shapes
: *
T0
|
flatten_1_input_1Placeholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
b
flatten_1_1/ShapeShapeflatten_1_input_1*
T0*
out_type0*
_output_shapes
:
i
flatten_1_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
[
flatten_1_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
^
flatten_1_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
flatten_1_1/ReshapeReshapeflatten_1_input_1flatten_1_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
o
dense_1_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_1_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�~ٽ
a
dense_1_1/random_uniform/maxConst*
valueB
 *�~�=*
dtype0*
_output_shapes
: 
�
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
dtype0*
_output_shapes
:	�*
seed2��h*
seed���)*
T0
�
dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1_1/random_uniform/mulMul&dense_1_1/random_uniform/RandomUniformdense_1_1/random_uniform/sub*
T0*
_output_shapes
:	�
�
dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
_output_shapes
:	�*
T0
�
dense_1_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�*
	container *
shape:	�
�
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	�
�
dense_1_1/kernel/readIdentitydense_1_1/kernel*
T0*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	�
^
dense_1_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
|
dense_1_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
�
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
use_locking(*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:�
x
dense_1_1/bias/readIdentitydense_1_1/bias*
_output_shapes	
:�*
T0*!
_class
loc:@dense_1_1/bias
�
dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
a
activation_1_1/ReluReludense_1_1/BiasAdd*(
_output_shapes
:����������*
T0
o
dense_2_1/random_uniform/shapeConst*
valueB"   d   *
dtype0*
_output_shapes
:
a
dense_2_1/random_uniform/minConst*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
a
dense_2_1/random_uniform/maxConst*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2Ӽ�*
seed���)
�
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
T0*
_output_shapes
:	�d
�
dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
T0*
_output_shapes
:	�d
�
dense_2_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
�
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*#
_class
loc:@dense_2_1/kernel
�
dense_2_1/kernel/readIdentitydense_2_1/kernel*
T0*#
_class
loc:@dense_2_1/kernel*
_output_shapes
:	�d
\
dense_2_1/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d
z
dense_2_1/bias
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
�
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
w
dense_2_1/bias/readIdentitydense_2_1/bias*
_output_shapes
:d*
T0*!
_class
loc:@dense_2_1/bias
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
�
dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������d*
T0
`
activation_2_1/ReluReludense_2_1/BiasAdd*'
_output_shapes
:���������d*
T0
o
dense_3_1/random_uniform/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
a
dense_3_1/random_uniform/minConst*
valueB
 *��L�*
dtype0*
_output_shapes
: 
a
dense_3_1/random_uniform/maxConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
_output_shapes

:d2*
seed2���*
seed���)*
T0*
dtype0
�
dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
T0*
_output_shapes

:d2
�
dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
T0*
_output_shapes

:d2
�
dense_3_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:d2*
	container *
shape
:d2
�
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2
�
dense_3_1/kernel/readIdentitydense_3_1/kernel*
T0*#
_class
loc:@dense_3_1/kernel*
_output_shapes

:d2
\
dense_3_1/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
z
dense_3_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:2*
	container *
shape:2
�
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0
w
dense_3_1/bias/readIdentitydense_3_1/bias*
T0*!
_class
loc:@dense_3_1/bias*
_output_shapes
:2
�
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( *
T0
�
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������2
`
activation_3_1/ReluReludense_3_1/BiasAdd*
T0*'
_output_shapes
:���������2
o
dense_4_1/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
a
dense_4_1/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
a
dense_4_1/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
&dense_4_1/random_uniform/RandomUniformRandomUniformdense_4_1/random_uniform/shape*
dtype0*
_output_shapes

:2*
seed2ֹM*
seed���)*
T0
�
dense_4_1/random_uniform/subSubdense_4_1/random_uniform/maxdense_4_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4_1/random_uniform/mulMul&dense_4_1/random_uniform/RandomUniformdense_4_1/random_uniform/sub*
T0*
_output_shapes

:2
�
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
_output_shapes

:2*
T0
�
dense_4_1/kernel
VariableV2*
dtype0*
_output_shapes

:2*
	container *
shape
:2*
shared_name 
�
dense_4_1/kernel/AssignAssigndense_4_1/kerneldense_4_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2
�
dense_4_1/kernel/readIdentitydense_4_1/kernel*#
_class
loc:@dense_4_1/kernel*
_output_shapes

:2*
T0
\
dense_4_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_4_1/bias
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
dense_4_1/bias/AssignAssigndense_4_1/biasdense_4_1/Const*
use_locking(*
T0*!
_class
loc:@dense_4_1/bias*
validate_shape(*
_output_shapes
:
w
dense_4_1/bias/readIdentitydense_4_1/bias*
T0*!
_class
loc:@dense_4_1/bias*
_output_shapes
:
�
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
dense_5_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
a
dense_5_1/random_uniform/minConst*
valueB
 *�m�*
dtype0*
_output_shapes
: 
a
dense_5_1/random_uniform/maxConst*
valueB
 *�m?*
dtype0*
_output_shapes
: 
�
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
dtype0*
_output_shapes

:*
seed2��*
seed���)*
T0
�
dense_5_1/random_uniform/subSubdense_5_1/random_uniform/maxdense_5_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_5_1/random_uniform/mulMul&dense_5_1/random_uniform/RandomUniformdense_5_1/random_uniform/sub*
T0*
_output_shapes

:
�
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
T0*
_output_shapes

:
�
dense_5_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:
�
dense_5_1/kernel/readIdentitydense_5_1/kernel*
T0*#
_class
loc:@dense_5_1/kernel*
_output_shapes

:
\
dense_5_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_5_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_5_1/bias/AssignAssigndense_5_1/biasdense_5_1/Const*
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
w
dense_5_1/bias/readIdentitydense_5_1/bias*
T0*!
_class
loc:@dense_5_1/bias*
_output_shapes
:
�
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
lambda_1_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*#
_output_shapes
:���������*
T0*
Index0
d
lambda_1_1/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
q
 lambda_1_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
s
"lambda_1_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*
T0*'
_output_shapes
:���������
q
 lambda_1_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
s
"lambda_1_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_slice_2StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
a
lambda_1_1/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
�
lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*
T0*
_output_shapes

:*

Tidx0*
	keep_dims(
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*
T0*'
_output_shapes
:���������
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
dtype0
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_output_shapes
: *
_class
loc:@dense_4/bias*
dtype0
�
IsVariableInitialized_8IsVariableInitializeddense_5/kernel*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_9IsVariableInitializeddense_5/bias*
_output_shapes
: *
_class
loc:@dense_5/bias*
dtype0
�
IsVariableInitialized_10IsVariableInitializedAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
{
IsVariableInitialized_11IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_14IsVariableInitialized
Adam/decay*
_output_shapes
: *
_class
loc:@Adam/decay*
dtype0
�
IsVariableInitialized_15IsVariableInitializeddense_1_1/kernel*#
_class
loc:@dense_1_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_16IsVariableInitializeddense_1_1/bias*
_output_shapes
: *!
_class
loc:@dense_1_1/bias*
dtype0
�
IsVariableInitialized_17IsVariableInitializeddense_2_1/kernel*#
_class
loc:@dense_2_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_18IsVariableInitializeddense_2_1/bias*!
_class
loc:@dense_2_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_19IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_20IsVariableInitializeddense_3_1/bias*!
_class
loc:@dense_3_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_21IsVariableInitializeddense_4_1/kernel*#
_class
loc:@dense_4_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_22IsVariableInitializeddense_4_1/bias*
_output_shapes
: *!
_class
loc:@dense_4_1/bias*
dtype0
�
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*#
_class
loc:@dense_5_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*!
_class
loc:@dense_5_1/bias*
dtype0*
_output_shapes
: 
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
dtype0*
_output_shapes
:	�*
shape:	�
�
AssignAssigndense_1_1/kernelPlaceholder*
_output_shapes
:	�*
use_locking( *
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(
X
Placeholder_1Placeholder*
_output_shapes	
:�*
shape:�*
dtype0
�
Assign_1Assigndense_1_1/biasPlaceholder_1*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:�*
use_locking( *
T0
`
Placeholder_2Placeholder*
_output_shapes
:	�d*
shape:	�d*
dtype0
�
Assign_2Assigndense_2_1/kernelPlaceholder_2*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking( *
T0
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
:d*
shape:d
�
Assign_3Assigndense_2_1/biasPlaceholder_3*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d
^
Placeholder_4Placeholder*
_output_shapes

:d2*
shape
:d2*
dtype0
�
Assign_4Assigndense_3_1/kernelPlaceholder_4*
use_locking( *
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
:2*
shape:2
�
Assign_5Assigndense_3_1/biasPlaceholder_5*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2
^
Placeholder_6Placeholder*
dtype0*
_output_shapes

:2*
shape
:2
�
Assign_6Assigndense_4_1/kernelPlaceholder_6*
use_locking( *
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2
V
Placeholder_7Placeholder*
dtype0*
_output_shapes
:*
shape:
�
Assign_7Assigndense_4_1/biasPlaceholder_7*
use_locking( *
T0*!
_class
loc:@dense_4_1/bias*
validate_shape(*
_output_shapes
:
^
Placeholder_8Placeholder*
shape
:*
dtype0*
_output_shapes

:
�
Assign_8Assigndense_5_1/kernelPlaceholder_8*
validate_shape(*
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@dense_5_1/kernel
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:*
shape:
�
Assign_9Assigndense_5_1/biasPlaceholder_9*
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(
^
SGD/iterations/initial_valueConst*
_output_shapes
: *
value	B	 R *
dtype0	
r
SGD/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: *
use_locking(
s
SGD/iterations/readIdentitySGD/iterations*
_output_shapes
: *
T0	*!
_class
loc:@SGD/iterations
Y
SGD/lr/initial_valueConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
j
SGD/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
[
SGD/lr/readIdentitySGD/lr*
T0*
_class
loc:@SGD/lr*
_output_shapes
: 
_
SGD/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 
m
SGD/momentum/readIdentitySGD/momentum*
T0*
_class
loc:@SGD/momentum*
_output_shapes
: 
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: *
use_locking(
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 
�
lambda_1_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
r
lambda_1_sample_weightsPlaceholder*#
_output_shapes
:���������*
shape:���������*
dtype0
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*
T0*'
_output_shapes
:���������
m
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*
T0*'
_output_shapes
:���������
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*
T0*#
_output_shapes
:���������
b
loss/lambda_1_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*#
_output_shapes
:���������*
T0
y
loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

b
loss/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:���������
d
loss/lambda_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
W
loss/mulMul
loss/mul/xloss/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
`
SGD_1/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
t
SGD_1/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD_1/iterations/AssignAssignSGD_1/iterationsSGD_1/iterations/initial_value*#
_class
loc:@SGD_1/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
y
SGD_1/iterations/readIdentitySGD_1/iterations*
T0	*#
_class
loc:@SGD_1/iterations*
_output_shapes
: 
[
SGD_1/lr/initial_valueConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
l
SGD_1/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: 
a
SGD_1/lr/readIdentitySGD_1/lr*
_output_shapes
: *
T0*
_class
loc:@SGD_1/lr
a
SGD_1/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
SGD_1/momentum
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
use_locking(*
T0*!
_class
loc:@SGD_1/momentum*
validate_shape(*
_output_shapes
: 
s
SGD_1/momentum/readIdentitySGD_1/momentum*
_output_shapes
: *
T0*!
_class
loc:@SGD_1/momentum
^
SGD_1/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
o
SGD_1/decay
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD_1/decay*
validate_shape(
j
SGD_1/decay/readIdentitySGD_1/decay*
T0*
_class
loc:@SGD_1/decay*
_output_shapes
: 
�
lambda_1_target_1Placeholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
t
lambda_1_sample_weights_1Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
r
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*
T0*'
_output_shapes
:���������
q
loss_1/lambda_1_loss/SquareSquareloss_1/lambda_1_loss/sub*
T0*'
_output_shapes
:���������
v
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
p
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB 
�
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
�
loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*#
_output_shapes
:���������*
T0
d
loss_1/lambda_1_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:���������
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
d
loss_1/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:���������
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Q
loss_1/mul/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
]

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
i
y_truePlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
g
maskPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
Y

loss_2/subSublambda_1/suby_true*
T0*'
_output_shapes
:���������
O

loss_2/AbsAbs
loss_2/sub*
T0*'
_output_shapes
:���������
R
loss_2/Less/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
`
loss_2/LessLess
loss_2/Absloss_2/Less/y*
T0*'
_output_shapes
:���������
U
loss_2/SquareSquare
loss_2/sub*
T0*'
_output_shapes
:���������
Q
loss_2/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
`

loss_2/mulMulloss_2/mul/xloss_2/Square*
T0*'
_output_shapes
:���������
Q
loss_2/Abs_1Abs
loss_2/sub*
T0*'
_output_shapes
:���������
S
loss_2/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
c
loss_2/sub_1Subloss_2/Abs_1loss_2/sub_1/y*
T0*'
_output_shapes
:���������
S
loss_2/mul_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*
T0*'
_output_shapes
:���������
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*
T0*'
_output_shapes
:���������
Z
loss_2/mul_2Mulloss_2/Selectmask*
T0*'
_output_shapes
:���������
g
loss_2/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
�

loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
�
loss_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
�
lambda_1_target_2Placeholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
n
loss_sample_weightsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
t
lambda_1_sample_weights_2Placeholder*#
_output_shapes
:���������*
shape:���������*
dtype0
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*
T0*#
_output_shapes
:���������
`
loss_3/loss_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*
T0*#
_output_shapes
:���������
u
loss_3/loss_loss/CastCastloss_3/loss_loss/NotEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

`
loss_3/loss_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*#
_output_shapes
:���������*
T0
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Q
loss_3/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y

loss_3/mulMulloss_3/mul/xloss_3/loss_loss/Mean_2*
T0*
_output_shapes
: 
l
loss_3/lambda_1_loss/zeros_like	ZerosLikelambda_1/sub*
T0*'
_output_shapes
:���������
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
�
loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*
T0*#
_output_shapes
:���������
d
loss_3/lambda_1_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:���������
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

d
loss_3/lambda_1_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*
T0*#
_output_shapes
:���������
f
loss_3/lambda_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
S
loss_3/mul_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
loss_3/mul_1Mulloss_3/mul_1/xloss_3/lambda_1_loss/Mean_2*
_output_shapes
: *
T0
L

loss_3/addAdd
loss_3/mulloss_3/mul_1*
T0*
_output_shapes
: 
{
!metrics_2/mean_absolute_error/subSublambda_1/sublambda_1_target_2*
T0*'
_output_shapes
:���������
}
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*'
_output_shapes
:���������*
T0

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
`
metrics_2/mean_q/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
metrics_2/mean_q/MeanMeanmetrics_2/mean_q/Maxmetrics_2/mean_q/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
[
metrics_2/mean_q/Const_1Const*
_output_shapes
: *
valueB *
dtype0
�
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
IsVariableInitialized_25IsVariableInitializedSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
y
IsVariableInitialized_26IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_27IsVariableInitializedSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitialized	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_29IsVariableInitializedSGD_1/iterations*#
_class
loc:@SGD_1/iterations*
dtype0	*
_output_shapes
: 
}
IsVariableInitialized_30IsVariableInitializedSGD_1/lr*
_class
loc:@SGD_1/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_31IsVariableInitializedSGD_1/momentum*!
_class
loc:@SGD_1/momentum*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_32IsVariableInitializedSGD_1/decay*
_class
loc:@SGD_1/decay*
dtype0*
_output_shapes
: 
�
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"_)��F      ��w	'�ǩ�AJ��
��
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
:
Less
x"T
y"T
z
"
Ttype:
2	
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
1
Square
x"T
y"T"
Ttype:

2	
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.6.02v1.6.0-0-gd2e24b6039��
z
flatten_1_inputPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
^
flatten_1/ShapeShapeflatten_1_input*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
i
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
\
flatten_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
���������
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*0
_output_shapes
:������������������*
T0*
Tshape0
m
dense_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
_
dense_1/random_uniform/minConst*
valueB
 *�~ٽ*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�~�=*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�*
seed2���
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes
:	�

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
_output_shapes
:	�*
T0
�
dense_1/kernel
VariableV2*
shape:	�*
shared_name *
dtype0*
_output_shapes
:	�*
	container 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
_output_shapes
:	�*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
z
dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_output_shapes	
:�*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:�
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
]
activation_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:����������
m
dense_2/random_uniform/shapeConst*
valueB"   d   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?��=
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�d*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	�d*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	�d*
T0
�
dense_2/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�d
|
dense_2/kernel/readIdentitydense_2/kernel*
_output_shapes
:	�d*
T0*!
_class
loc:@dense_2/kernel
Z
dense_2/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:d*
	container *
shape:d*
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:d
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:d
�
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
\
activation_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:���������d
m
dense_3/random_uniform/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
_
dense_3/random_uniform/minConst*
_output_shapes
: *
valueB
 *��L�*
dtype0
_
dense_3/random_uniform/maxConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:d2*
seed2�ځ
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:d2
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:d2*
T0
�
dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking(
{
dense_3/kernel/readIdentitydense_3/kernel*
_output_shapes

:d2*
T0*!
_class
loc:@dense_3/kernel
Z
dense_3/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
x
dense_3/bias
VariableV2*
dtype0*
_output_shapes
:2*
	container *
shape:2*
shared_name 
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:2
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
_output_shapes
:2*
T0
�
dense_3/MatMulMatMulactivation_2/Reludense_3/kernel/read*
T0*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( 
�
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:���������2*
T0
\
activation_3/ReluReludense_3/BiasAdd*
T0*'
_output_shapes
:���������2
m
dense_4/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
_
dense_4/random_uniform/minConst*
_output_shapes
: *
valueB
 *�D��*
dtype0
_
dense_4/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
dtype0*
_output_shapes

:2*
seed2��K*
seed���)*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

:2*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:2*
T0
�
dense_4/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:2*
	container *
shape
:2
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
validate_shape(*
_output_shapes

:2*
use_locking(*
T0*!
_class
loc:@dense_4/kernel
{
dense_4/kernel/readIdentitydense_4/kernel*
T0*!
_class
loc:@dense_4/kernel*
_output_shapes

:2
Z
dense_4/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_4/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:
q
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0
�
dense_4/MatMulMatMulactivation_3/Reludense_4/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
d
activation_4/IdentityIdentitydense_4/BiasAdd*'
_output_shapes
:���������*
T0
m
dense_5/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
valueB
 *�m�*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
valueB
 *�m?*
dtype0*
_output_shapes
: 
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
dtype0*
_output_shapes

:*
seed2��*
seed���)*
T0
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
_output_shapes
: *
T0
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes

:
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes

:
�
dense_5/kernel
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:
{
dense_5/kernel/readIdentitydense_5/kernel*
_output_shapes

:*
T0*!
_class
loc:@dense_5/kernel
Z
dense_5/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_5/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
q
dense_5/bias/readIdentitydense_5/bias*
_output_shapes
:*
T0*
_class
loc:@dense_5/bias
�
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
m
lambda_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:���������
b
lambda_1/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
lambda_1/ExpandDims
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
o
lambda_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*'
_output_shapes
:���������*
T0
o
lambda_1/strided_slice_2/stackConst*
_output_shapes
:*
valueB"       *
dtype0
q
 lambda_1/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
_
lambda_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:
b
lambda_1/subSublambda_1/addlambda_1/Mean*'
_output_shapes
:���������*
T0
_
Adam/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*"
_class
loc:@Adam/iterations*
_output_shapes
: *
T0	
Z
Adam/lr/initial_valueConst*
valueB
 *o�9*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/lr
^
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
_output_shapes
: *
T0
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
valueB
 *w�?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
use_locking(*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
_output_shapes
: *
T0
|
flatten_1_input_1Placeholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
b
flatten_1_1/ShapeShapeflatten_1_input_1*
T0*
out_type0*
_output_shapes
:
i
flatten_1_1/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
k
!flatten_1_1/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
[
flatten_1_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
^
flatten_1_1/stack/0Const*
_output_shapes
: *
valueB :
���������*
dtype0
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
T0*

axis *
N*
_output_shapes
:
�
flatten_1_1/ReshapeReshapeflatten_1_input_1flatten_1_1/stack*
T0*
Tshape0*0
_output_shapes
:������������������
o
dense_1_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
a
dense_1_1/random_uniform/minConst*
valueB
 *�~ٽ*
dtype0*
_output_shapes
: 
a
dense_1_1/random_uniform/maxConst*
valueB
 *�~�=*
dtype0*
_output_shapes
: 
�
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
dtype0*
_output_shapes
:	�*
seed2��h*
seed���)*
T0
�
dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1_1/random_uniform/mulMul&dense_1_1/random_uniform/RandomUniformdense_1_1/random_uniform/sub*
_output_shapes
:	�*
T0
�
dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
_output_shapes
:	�*
T0
�
dense_1_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�*
	container *
shape:	�
�
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	�
�
dense_1_1/kernel/readIdentitydense_1_1/kernel*
T0*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	�
^
dense_1_1/ConstConst*
_output_shapes	
:�*
valueB�*    *
dtype0
|
dense_1_1/bias
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
�
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
use_locking(*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:�
x
dense_1_1/bias/readIdentitydense_1_1/bias*!
_class
loc:@dense_1_1/bias*
_output_shapes	
:�*
T0
�
dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
a
activation_1_1/ReluReludense_1_1/BiasAdd*
T0*(
_output_shapes
:����������
o
dense_2_1/random_uniform/shapeConst*
valueB"   d   *
dtype0*
_output_shapes
:
a
dense_2_1/random_uniform/minConst*
valueB
 *?�ʽ*
dtype0*
_output_shapes
: 
a
dense_2_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?��=
�
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2Ӽ�*
seed���)
�
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
T0*
_output_shapes
:	�d
�
dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
T0*
_output_shapes
:	�d
�
dense_2_1/kernel
VariableV2*
shape:	�d*
shared_name *
dtype0*
_output_shapes
:	�d*
	container 
�
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(
�
dense_2_1/kernel/readIdentitydense_2_1/kernel*
T0*#
_class
loc:@dense_2_1/kernel*
_output_shapes
:	�d
\
dense_2_1/ConstConst*
valueBd*    *
dtype0*
_output_shapes
:d
z
dense_2_1/bias
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
�
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
use_locking(*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d
w
dense_2_1/bias/readIdentitydense_2_1/bias*!
_class
loc:@dense_2_1/bias*
_output_shapes
:d*
T0
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
�
dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������d*
T0
`
activation_2_1/ReluReludense_2_1/BiasAdd*
T0*'
_output_shapes
:���������d
o
dense_3_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"d   2   
a
dense_3_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *��L�
a
dense_3_1/random_uniform/maxConst*
valueB
 *��L>*
dtype0*
_output_shapes
: 
�
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:d2*
seed2���
�
dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
T0*
_output_shapes

:d2
�
dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
T0*
_output_shapes

:d2
�
dense_3_1/kernel
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
�
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
_output_shapes

:d2*
use_locking(*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(
�
dense_3_1/kernel/readIdentitydense_3_1/kernel*
T0*#
_class
loc:@dense_3_1/kernel*
_output_shapes

:d2
\
dense_3_1/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
z
dense_3_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:2*
	container *
shape:2
�
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2
w
dense_3_1/bias/readIdentitydense_3_1/bias*
_output_shapes
:2*
T0*!
_class
loc:@dense_3_1/bias
�
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������2*
transpose_a( 
�
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������2*
T0
`
activation_3_1/ReluReludense_3_1/BiasAdd*'
_output_shapes
:���������2*
T0
o
dense_4_1/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
a
dense_4_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *�D��*
dtype0
a
dense_4_1/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
&dense_4_1/random_uniform/RandomUniformRandomUniformdense_4_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:2*
seed2ֹM
�
dense_4_1/random_uniform/subSubdense_4_1/random_uniform/maxdense_4_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_4_1/random_uniform/mulMul&dense_4_1/random_uniform/RandomUniformdense_4_1/random_uniform/sub*
T0*
_output_shapes

:2
�
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
T0*
_output_shapes

:2
�
dense_4_1/kernel
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
�
dense_4_1/kernel/AssignAssigndense_4_1/kerneldense_4_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2
�
dense_4_1/kernel/readIdentitydense_4_1/kernel*#
_class
loc:@dense_4_1/kernel*
_output_shapes

:2*
T0
\
dense_4_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_4_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_4_1/bias/AssignAssigndense_4_1/biasdense_4_1/Const*
T0*!
_class
loc:@dense_4_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
w
dense_4_1/bias/readIdentitydense_4_1/bias*
T0*!
_class
loc:@dense_4_1/bias*
_output_shapes
:
�
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
o
dense_5_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_5_1/random_uniform/minConst*
valueB
 *�m�*
dtype0*
_output_shapes
: 
a
dense_5_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�m?*
dtype0
�
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:*
seed2��
�
dense_5_1/random_uniform/subSubdense_5_1/random_uniform/maxdense_5_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_5_1/random_uniform/mulMul&dense_5_1/random_uniform/RandomUniformdense_5_1/random_uniform/sub*
T0*
_output_shapes

:
�
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
T0*
_output_shapes

:
�
dense_5_1/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@dense_5_1/kernel
�
dense_5_1/kernel/readIdentitydense_5_1/kernel*#
_class
loc:@dense_5_1/kernel*
_output_shapes

:*
T0
\
dense_5_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_5_1/bias
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
dense_5_1/bias/AssignAssigndense_5_1/biasdense_5_1/Const*
use_locking(*
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:
w
dense_5_1/bias/readIdentitydense_5_1/bias*
T0*!
_class
loc:@dense_5_1/bias*
_output_shapes
:
�
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
o
lambda_1_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*#
_output_shapes
:���������*
T0*
Index0
d
lambda_1_1/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
q
 lambda_1_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"       
s
"lambda_1_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*
T0*'
_output_shapes
:���������
q
 lambda_1_1/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_2/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
s
"lambda_1_1/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
lambda_1_1/strided_slice_2StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
a
lambda_1_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*
T0*'
_output_shapes
:���������
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializeddense_5/kernel*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_9IsVariableInitializeddense_5/bias*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
{
IsVariableInitialized_11IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_14IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_15IsVariableInitializeddense_1_1/kernel*#
_class
loc:@dense_1_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_16IsVariableInitializeddense_1_1/bias*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1_1/bias
�
IsVariableInitialized_17IsVariableInitializeddense_2_1/kernel*
dtype0*
_output_shapes
: *#
_class
loc:@dense_2_1/kernel
�
IsVariableInitialized_18IsVariableInitializeddense_2_1/bias*!
_class
loc:@dense_2_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_19IsVariableInitializeddense_3_1/kernel*
dtype0*
_output_shapes
: *#
_class
loc:@dense_3_1/kernel
�
IsVariableInitialized_20IsVariableInitializeddense_3_1/bias*!
_class
loc:@dense_3_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_21IsVariableInitializeddense_4_1/kernel*#
_class
loc:@dense_4_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_22IsVariableInitializeddense_4_1/bias*!
_class
loc:@dense_4_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*
_output_shapes
: *#
_class
loc:@dense_5_1/kernel*
dtype0
�
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*!
_class
loc:@dense_5_1/bias*
dtype0*
_output_shapes
: 
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
shape:	�*
dtype0*
_output_shapes
:	�
�
AssignAssigndense_1_1/kernelPlaceholder*
use_locking( *
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	�
X
Placeholder_1Placeholder*
dtype0*
_output_shapes	
:�*
shape:�
�
Assign_1Assigndense_1_1/biasPlaceholder_1*
validate_shape(*
_output_shapes	
:�*
use_locking( *
T0*!
_class
loc:@dense_1_1/bias
`
Placeholder_2Placeholder*
_output_shapes
:	�d*
shape:	�d*
dtype0
�
Assign_2Assigndense_2_1/kernelPlaceholder_2*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking( *
T0
V
Placeholder_3Placeholder*
shape:d*
dtype0*
_output_shapes
:d
�
Assign_3Assigndense_2_1/biasPlaceholder_3*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d
^
Placeholder_4Placeholder*
dtype0*
_output_shapes

:d2*
shape
:d2
�
Assign_4Assigndense_3_1/kernelPlaceholder_4*
use_locking( *
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
:2*
shape:2
�
Assign_5Assigndense_3_1/biasPlaceholder_5*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2
^
Placeholder_6Placeholder*
_output_shapes

:2*
shape
:2*
dtype0
�
Assign_6Assigndense_4_1/kernelPlaceholder_6*
use_locking( *
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2
V
Placeholder_7Placeholder*
shape:*
dtype0*
_output_shapes
:
�
Assign_7Assigndense_4_1/biasPlaceholder_7*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@dense_4_1/bias
^
Placeholder_8Placeholder*
dtype0*
_output_shapes

:*
shape
:
�
Assign_8Assigndense_5_1/kernelPlaceholder_8*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking( 
V
Placeholder_9Placeholder*
shape:*
dtype0*
_output_shapes
:
�
Assign_9Assigndense_5_1/biasPlaceholder_9*
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
r
SGD/iterations
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0	
�
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
use_locking(*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: 
s
SGD/iterations/readIdentitySGD/iterations*
_output_shapes
: *
T0	*!
_class
loc:@SGD/iterations
Y
SGD/lr/initial_valueConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
j
SGD/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
[
SGD/lr/readIdentitySGD/lr*
T0*
_class
loc:@SGD/lr*
_output_shapes
: 
_
SGD/momentum/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
p
SGD/momentum
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/momentum
m
SGD/momentum/readIdentitySGD/momentum*
_output_shapes
: *
T0*
_class
loc:@SGD/momentum
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: *
use_locking(
d
SGD/decay/readIdentity	SGD/decay*
_class
loc:@SGD/decay*
_output_shapes
: *
T0
�
lambda_1_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
r
lambda_1_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*
T0*'
_output_shapes
:���������
m
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*
T0*'
_output_shapes
:���������
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*
T0*#
_output_shapes
:���������
b
loss/lambda_1_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:���������
y
loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
b
loss/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*#
_output_shapes
:���������*
T0
d
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
`
SGD_1/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
t
SGD_1/iterations
VariableV2*
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
�
SGD_1/iterations/AssignAssignSGD_1/iterationsSGD_1/iterations/initial_value*
use_locking(*
T0	*#
_class
loc:@SGD_1/iterations*
validate_shape(*
_output_shapes
: 
y
SGD_1/iterations/readIdentitySGD_1/iterations*
T0	*#
_class
loc:@SGD_1/iterations*
_output_shapes
: 
[
SGD_1/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
l
SGD_1/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: 
a
SGD_1/lr/readIdentitySGD_1/lr*
T0*
_class
loc:@SGD_1/lr*
_output_shapes
: 
a
SGD_1/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
SGD_1/momentum
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
use_locking(*
T0*!
_class
loc:@SGD_1/momentum*
validate_shape(*
_output_shapes
: 
s
SGD_1/momentum/readIdentitySGD_1/momentum*
T0*!
_class
loc:@SGD_1/momentum*
_output_shapes
: 
^
SGD_1/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
SGD_1/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD_1/decay
j
SGD_1/decay/readIdentitySGD_1/decay*
T0*
_class
loc:@SGD_1/decay*
_output_shapes
: 
�
lambda_1_target_1Placeholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
t
lambda_1_sample_weights_1Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
r
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*'
_output_shapes
:���������*
T0
q
loss_1/lambda_1_loss/SquareSquareloss_1/lambda_1_loss/sub*'
_output_shapes
:���������*
T0
v
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
p
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
�
loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*
T0*#
_output_shapes
:���������
d
loss_1/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*#
_output_shapes
:���������*
T0
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

d
loss_1/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:���������
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Q
loss_1/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
]

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
i
y_truePlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
g
maskPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
Y

loss_2/subSublambda_1/suby_true*
T0*'
_output_shapes
:���������
O

loss_2/AbsAbs
loss_2/sub*
T0*'
_output_shapes
:���������
R
loss_2/Less/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
`
loss_2/LessLess
loss_2/Absloss_2/Less/y*'
_output_shapes
:���������*
T0
U
loss_2/SquareSquare
loss_2/sub*
T0*'
_output_shapes
:���������
Q
loss_2/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
`

loss_2/mulMulloss_2/mul/xloss_2/Square*'
_output_shapes
:���������*
T0
Q
loss_2/Abs_1Abs
loss_2/sub*
T0*'
_output_shapes
:���������
S
loss_2/sub_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
c
loss_2/sub_1Subloss_2/Abs_1loss_2/sub_1/y*'
_output_shapes
:���������*
T0
S
loss_2/mul_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*
T0*'
_output_shapes
:���������
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*
T0*'
_output_shapes
:���������
Z
loss_2/mul_2Mulloss_2/Selectmask*'
_output_shapes
:���������*
T0
g
loss_2/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�

loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
�
loss_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
�
lambda_1_target_2Placeholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
n
loss_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
t
lambda_1_sample_weights_2Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*
T0*#
_output_shapes
:���������
`
loss_3/loss_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*
T0*#
_output_shapes
:���������
u
loss_3/loss_loss/CastCastloss_3/loss_loss/NotEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
`
loss_3/loss_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*#
_output_shapes
:���������*
T0
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Q
loss_3/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
Y

loss_3/mulMulloss_3/mul/xloss_3/loss_loss/Mean_2*
T0*
_output_shapes
: 
l
loss_3/lambda_1_loss/zeros_like	ZerosLikelambda_1/sub*
T0*'
_output_shapes
:���������
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
�
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
�
loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*
T0*#
_output_shapes
:���������
d
loss_3/lambda_1_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:���������
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
d
loss_3/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*
T0*#
_output_shapes
:���������
f
loss_3/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
S
loss_3/mul_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
a
loss_3/mul_1Mulloss_3/mul_1/xloss_3/lambda_1_loss/Mean_2*
T0*
_output_shapes
: 
L

loss_3/addAdd
loss_3/mulloss_3/mul_1*
T0*
_output_shapes
: 
{
!metrics_2/mean_absolute_error/subSublambda_1/sublambda_1_target_2*
T0*'
_output_shapes
:���������
}
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*
T0*'
_output_shapes
:���������

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
`
metrics_2/mean_q/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
metrics_2/mean_q/MeanMeanmetrics_2/mean_q/Maxmetrics_2/mean_q/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
[
metrics_2/mean_q/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
�
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
IsVariableInitialized_25IsVariableInitializedSGD/iterations*
dtype0	*
_output_shapes
: *!
_class
loc:@SGD/iterations
y
IsVariableInitialized_26IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_27IsVariableInitializedSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitialized	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_29IsVariableInitializedSGD_1/iterations*#
_class
loc:@SGD_1/iterations*
dtype0	*
_output_shapes
: 
}
IsVariableInitialized_30IsVariableInitializedSGD_1/lr*
_class
loc:@SGD_1/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_31IsVariableInitializedSGD_1/momentum*
_output_shapes
: *!
_class
loc:@SGD_1/momentum*
dtype0
�
IsVariableInitialized_32IsVariableInitializedSGD_1/decay*
_class
loc:@SGD_1/decay*
dtype0*
_output_shapes
: 
�
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign""�
trainable_variables��
Z
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:0
K
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:0
Z
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:0
K
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:0
Z
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:0
K
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:0
Z
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02dense_4/random_uniform:0
K
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02dense_4/Const:0
Z
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02dense_5/random_uniform:0
K
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02dense_5/Const:0
d
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:0
D
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:0
T
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:0
T
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:0
P
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:0
b
dense_1_1/kernel:0dense_1_1/kernel/Assigndense_1_1/kernel/read:02dense_1_1/random_uniform:0
S
dense_1_1/bias:0dense_1_1/bias/Assigndense_1_1/bias/read:02dense_1_1/Const:0
b
dense_2_1/kernel:0dense_2_1/kernel/Assigndense_2_1/kernel/read:02dense_2_1/random_uniform:0
S
dense_2_1/bias:0dense_2_1/bias/Assigndense_2_1/bias/read:02dense_2_1/Const:0
b
dense_3_1/kernel:0dense_3_1/kernel/Assigndense_3_1/kernel/read:02dense_3_1/random_uniform:0
S
dense_3_1/bias:0dense_3_1/bias/Assigndense_3_1/bias/read:02dense_3_1/Const:0
b
dense_4_1/kernel:0dense_4_1/kernel/Assigndense_4_1/kernel/read:02dense_4_1/random_uniform:0
S
dense_4_1/bias:0dense_4_1/bias/Assigndense_4_1/bias/read:02dense_4_1/Const:0
b
dense_5_1/kernel:0dense_5_1/kernel/Assigndense_5_1/kernel/read:02dense_5_1/random_uniform:0
S
dense_5_1/bias:0dense_5_1/bias/Assigndense_5_1/bias/read:02dense_5_1/Const:0
`
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:0
@
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:0
X
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:0
L
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:0
h
SGD_1/iterations:0SGD_1/iterations/AssignSGD_1/iterations/read:02 SGD_1/iterations/initial_value:0
H

SGD_1/lr:0SGD_1/lr/AssignSGD_1/lr/read:02SGD_1/lr/initial_value:0
`
SGD_1/momentum:0SGD_1/momentum/AssignSGD_1/momentum/read:02SGD_1/momentum/initial_value:0
T
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0"�
	variables��
Z
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:0
K
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:0
Z
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:0
K
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:0
Z
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02dense_3/random_uniform:0
K
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02dense_3/Const:0
Z
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02dense_4/random_uniform:0
K
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02dense_4/Const:0
Z
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02dense_5/random_uniform:0
K
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02dense_5/Const:0
d
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:0
D
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:0
T
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:0
T
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:0
P
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:0
b
dense_1_1/kernel:0dense_1_1/kernel/Assigndense_1_1/kernel/read:02dense_1_1/random_uniform:0
S
dense_1_1/bias:0dense_1_1/bias/Assigndense_1_1/bias/read:02dense_1_1/Const:0
b
dense_2_1/kernel:0dense_2_1/kernel/Assigndense_2_1/kernel/read:02dense_2_1/random_uniform:0
S
dense_2_1/bias:0dense_2_1/bias/Assigndense_2_1/bias/read:02dense_2_1/Const:0
b
dense_3_1/kernel:0dense_3_1/kernel/Assigndense_3_1/kernel/read:02dense_3_1/random_uniform:0
S
dense_3_1/bias:0dense_3_1/bias/Assigndense_3_1/bias/read:02dense_3_1/Const:0
b
dense_4_1/kernel:0dense_4_1/kernel/Assigndense_4_1/kernel/read:02dense_4_1/random_uniform:0
S
dense_4_1/bias:0dense_4_1/bias/Assigndense_4_1/bias/read:02dense_4_1/Const:0
b
dense_5_1/kernel:0dense_5_1/kernel/Assigndense_5_1/kernel/read:02dense_5_1/random_uniform:0
S
dense_5_1/bias:0dense_5_1/bias/Assigndense_5_1/bias/read:02dense_5_1/Const:0
`
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:0
@
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:0
X
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:0
L
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:0
h
SGD_1/iterations:0SGD_1/iterations/AssignSGD_1/iterations/read:02 SGD_1/iterations/initial_value:0
H

SGD_1/lr:0SGD_1/lr/AssignSGD_1/lr/read:02SGD_1/lr/initial_value:0
`
SGD_1/momentum:0SGD_1/momentum/AssignSGD_1/momentum/read:02SGD_1/momentum/initial_value:0
T
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0�Hb�.       ��W�	>B��ǩ�A*#
!
Average reward per episodeO�����,       ���E	�B��ǩ�A*!

total reward per episode  ��w<�-       <A��	u9��ǩ�A* 

Average reward per stepO��ۮfk       `/�#	:��ǩ�A*

epsilonO���<�-       <A��	tb��ǩ�A* 

Average reward per stepO���s�       `/�#	^c��ǩ�A*

epsilonO���/i�-       <A��	d���ǩ�A* 

Average reward per stepO�����       `/�#	S���ǩ�A*

epsilonO���U-       <A��	���ǩ�A* 

Average reward per stepO���U2�       `/�#	���ǩ�A*

epsilonO���/b-       <A��	����ǩ�A* 

Average reward per stepO��)��H       `/�#	b���ǩ�A*

epsilonO��,��~-       <A��	R���ǩ�A* 

Average reward per stepO���}\(       `/�#	���ǩ�A*

epsilonO���H��-       <A��	{���ǩ�A * 

Average reward per stepO��ײ�       `/�#	#���ǩ�A *

epsilonO��0�_-       <A��	殗�ǩ�A!* 

Average reward per stepO��+iW       `/�#	����ǩ�A!*

epsilonO��rP��-       <A��	���ǩ�A"* 

Average reward per stepO���˜�       `/�#	����ǩ�A"*

epsilonO��B� -       <A��	R��ǩ�A#* 

Average reward per stepO���Ć�       `/�#	 S��ǩ�A#*

epsilonO���4�C-       <A��	���ǩ�A$* 

Average reward per stepO���v��       `/�#	����ǩ�A$*

epsilonO���4$�-       <A��	���ǩ�A%* 

Average reward per stepO��Vn�       `/�#	ٲ��ǩ�A%*

epsilonO���5-       <A��	8��ǩ�A&* 

Average reward per stepO������       `/�#	��ǩ�A&*

epsilonO��pa��-       <A��	s.��ǩ�A'* 

Average reward per stepO��7O       `/�#	/��ǩ�A'*

epsilonO��R�n-       <A��	�	��ǩ�A(* 

Average reward per stepO���Տa       `/�#	g
��ǩ�A(*

epsilonO��ej�-       <A��	�]��ǩ�A)* 

Average reward per stepO���,�C       `/�#	�^��ǩ�A)*

epsilonO��Ԏ^a-       <A��	����ǩ�A** 

Average reward per stepO��4C��       `/�#	@���ǩ�A**

epsilonO��E��h-       <A��	f���ǩ�A+* 

Average reward per stepO��O�J�       `/�#	����ǩ�A+*

epsilonO��&W��-       <A��	/���ǩ�A,* 

Average reward per stepO��2%%       `/�#	닱�ǩ�A,*

epsilonO��=���-       <A��	�z��ǩ�A-* 

Average reward per stepO���_�&       `/�#	x{��ǩ�A-*

epsilonO�����-       <A��	���ǩ�A.* 

Average reward per stepO��i�       `/�#	Z���ǩ�A.*

epsilonO��~1�}-       <A��	�Ƕ�ǩ�A/* 

Average reward per stepO��ߙ�
       `/�#	Cȶ�ǩ�A/*

epsilonO����\,-       <A��	�Ҹ�ǩ�A0* 

Average reward per stepO�����       `/�#	Ӹ�ǩ�A0*

epsilonO��^J��-       <A��	J��ǩ�A1* 

Average reward per stepO��G�5       `/�#	���ǩ�A1*

epsilonO��[G_�-       <A��	�ۼ�ǩ�A2* 

Average reward per stepO��(�#�       `/�#	rܼ�ǩ�A2*

epsilonO������-       <A��	�о�ǩ�A3* 

Average reward per stepO���
R�       `/�#	�Ѿ�ǩ�A3*

epsilonO��b�-       <A��	d���ǩ�A4* 

Average reward per stepO���Ai,       `/�#	>���ǩ�A4*

epsilonO��HQ(�-       <A��	����ǩ�A5* 

Average reward per stepO���au�       `/�#	����ǩ�A5*

epsilonO���F-       <A��	[���ǩ�A6* 

Average reward per stepO��:\�{       `/�#	���ǩ�A6*

epsilonO��^4�-       <A��	e���ǩ�A7* 

Average reward per stepO���%o�       `/�#	!���ǩ�A7*

epsilonO��jzt-       <A��	.��ǩ�A8* 

Average reward per stepO���y�       `/�#	���ǩ�A8*

epsilonO���.�K-       <A��	 ��ǩ�A9* 

Average reward per stepO���%�8       `/�#	���ǩ�A9*

epsilonO���SL�-       <A��	���ǩ�A:* 

Average reward per stepO�����o       `/�#	���ǩ�A:*

epsilonO����e�-       <A��	�#��ǩ�A;* 

Average reward per stepO�����       `/�#	1$��ǩ�A;*

epsilonO���`�-       <A��	�*��ǩ�A<* 

Average reward per stepO��GsL�       `/�#	g+��ǩ�A<*

epsilonO����5-       <A��	R(��ǩ�A=* 

Average reward per stepO��:�,       `/�#	N*��ǩ�A=*

epsilonO���m)-       <A��	�6��ǩ�A>* 

Average reward per stepO�����O       `/�#	7��ǩ�A>*

epsilonO��1�-       <A��	�!��ǩ�A?* 

Average reward per stepO��v       `/�#	�"��ǩ�A?*

epsilonO����[�-       <A��	�(��ǩ�A@* 

Average reward per stepO��n���       `/�#	�)��ǩ�A@*

epsilonO��g�)-       <A��	���ǩ�AA* 

Average reward per stepO��`m�_       `/�#	H��ǩ�AA*

epsilonO��Sb-       <A��	%t��ǩ�AB* 

Average reward per stepO��`��T       `/�#	�t��ǩ�AB*

epsilonO���JeD-       <A��	*r��ǩ�AC* 

Average reward per stepO��k��n       `/�#	�r��ǩ�AC*

epsilonO��:�-       <A��	5]��ǩ�AD* 

Average reward per stepO����.J       `/�#	�]��ǩ�AD*

epsilonO�����-       <A��	�X��ǩ�AE* 

Average reward per stepO���嫱       `/�#	.Y��ǩ�AE*

epsilonO��\��-       <A��	 E��ǩ�AF* 

Average reward per stepO��@�       `/�#	�E��ǩ�AF*

epsilonO����
�-       <A��	�<��ǩ�AG* 

Average reward per stepO����0-       `/�#	C=��ǩ�AG*

epsilonO��Cҏ�-       <A��	Z~��ǩ�AH* 

Average reward per stepO��.�E/       `/�#	�~��ǩ�AH*

epsilonO��_���-       <A��	�p��ǩ�AI* 

Average reward per stepO��B�s6       `/�#	qq��ǩ�AI*

epsilonO���>a-       <A��	�^��ǩ�AJ* 

Average reward per stepO�� �TI       `/�#	F_��ǩ�AJ*

epsilonO��Cg�-       <A��	ؼ��ǩ�AK* 

Average reward per stepO����J�       `/�#	{���ǩ�AK*

epsilonO��D�p�-       <A��	g���ǩ�AL* 

Average reward per stepO��OQ�       `/�#	A���ǩ�AL*

epsilonO��b�^�-       <A��	����ǩ�AM* 

Average reward per stepO��s9w       `/�#	P ��ǩ�AM*

epsilonO��΁�<-       <A��	����ǩ�AN* 

Average reward per stepO��y�Ӣ       `/�#	���ǩ�AN*

epsilonO���F�-       <A��	�&��ǩ�AO* 

Average reward per stepO���=Σ       `/�#	g'��ǩ�AO*

epsilonO��M�:<-       <A��	���ǩ�AP* 

Average reward per stepO��r�(p       `/�#	H��ǩ�AP*

epsilonO��:f��-       <A��	���ǩ�AQ* 

Average reward per stepO��l���       `/�#	W��ǩ�AQ*

epsilonO���C��-       <A��	8���ǩ�AR* 

Average reward per stepO���B�        `/�#	����ǩ�AR*

epsilonO�����-       <A��	�@��ǩ�AS* 

Average reward per stepO���7S       `/�#	xA��ǩ�AS*

epsilonO���-       <A��	U4��ǩ�AT* 

Average reward per stepO��l��       `/�#	5��ǩ�AT*

epsilonO���s�-       <A��	�$��ǩ�AU* 

Average reward per stepO��I���       `/�#	�%��ǩ�AU*

epsilonO����41-       <A��	  �ǩ�AV* 

Average reward per stepO��2Wo       `/�#	� �ǩ�AV*

epsilonO���>��-       <A��	8f�ǩ�AW* 

Average reward per stepO���+A       `/�#	�f�ǩ�AW*

epsilonO����rV-       <A��	P�ǩ�AX* 

Average reward per stepO����M�       `/�#	�P�ǩ�AX*

epsilonO��� -       <A��	z6�ǩ�AY* 

Average reward per stepO��5�"�       `/�#	7�ǩ�AY*

epsilonO��hQ�T-       <A��	��ǩ�AZ* 

Average reward per stepO���Zs7       `/�#	��ǩ�AZ*

epsilonO���e-       <A��	�w�ǩ�A[* 

Average reward per stepO������       `/�#	�x�ǩ�A[*

epsilonO��6��-       <A��	\r
�ǩ�A\* 

Average reward per stepO���AsX       `/�#	�r
�ǩ�A\*

epsilonO������-       <A��	0c�ǩ�A]* 

Average reward per stepO�����       `/�#	�c�ǩ�A]*

epsilonO���o�o-       <A��	ga�ǩ�A^* 

Average reward per stepO����<       `/�#	�a�ǩ�A^*

epsilonO���/�-       <A��	[�ǩ�A_* 

Average reward per stepO��Ȑ�       `/�#	�[�ǩ�A_*

epsilonO��L�Xc0       ���_	x�ǩ�A*#
!
Average reward per episode��:�Q���.       ��W�	
��ǩ�A*!

total reward per episode  L����-       <A��	��ǩ�A`* 

Average reward per step��:�I!�)       `/�#	���ǩ�A`*

epsilon��:��Pi�-       <A��	C��ǩ�Aa* 

Average reward per step��:��tޭ       `/�#	��ǩ�Aa*

epsilon��:�פ��-       <A��	?r�ǩ�Ab* 

Average reward per step��:��h�P       `/�#	�r�ǩ�Ab*

epsilon��:�@$ �-       <A��	���ǩ�Ac* 

Average reward per step��:��JF       `/�#	��ǩ�Ac*

epsilon��:�UX͘-       <A��	\��ǩ�Ad* 

Average reward per step��:��l(�       `/�#	��ǩ�Ad*

epsilon��:�g���-       <A��	)��ǩ�Ae* 

Average reward per step��:��cV       `/�#	���ǩ�Ae*

epsilon��:�F �-       <A��	��ǩ�Af* 

Average reward per step��:��:T       `/�#	���ǩ�Af*

epsilon��:����-       <A��	� �ǩ�Ag* 

Average reward per step��:��F�       `/�#	�� �ǩ�Ag*

epsilon��:�A��.-       <A��	�"�ǩ�Ah* 

Average reward per step��:����       `/�#	��"�ǩ�Ah*

epsilon��:���?-       <A��	R�$�ǩ�Ai* 

Average reward per step��:�&Mu<       `/�#	��$�ǩ�Ai*

epsilon��:�)>��-       <A��	��&�ǩ�Aj* 

Average reward per step��:��WK�       `/�#	��&�ǩ�Aj*

epsilon��:��Z{b-       <A��	��(�ǩ�Ak* 

Average reward per step��:��,V       `/�#	M�(�ǩ�Ak*

epsilon��:��Ӣ2-       <A��	�0+�ǩ�Al* 

Average reward per step��:��+�       `/�#	Y1+�ǩ�Al*

epsilon��:�b\֎-       <A��	�X-�ǩ�Am* 

Average reward per step��:�l[�       `/�#	\Y-�ǩ�Am*

epsilon��:�s+�-       <A��	�[/�ǩ�An* 

Average reward per step��:��ES       `/�#	�\/�ǩ�An*

epsilon��:�t��p-       <A��	�Q1�ǩ�Ao* 

Average reward per step��:��]��       `/�#	�R1�ǩ�Ao*

epsilon��:���b2-       <A��	�o3�ǩ�Ap* 

Average reward per step��:��"d       `/�#	�p3�ǩ�Ap*

epsilon��:�+��;-       <A��	�d5�ǩ�Aq* 

Average reward per step��:�U�n       `/�#	(e5�ǩ�Aq*

epsilon��:�C��-       <A��	m7�ǩ�Ar* 

Average reward per step��:��3�       `/�#	�m7�ǩ�Ar*

epsilon��:�|~f�-       <A��	�T9�ǩ�As* 

Average reward per step��:���       `/�#	XU9�ǩ�As*

epsilon��:�I|?-       <A��	�O;�ǩ�At* 

Average reward per step��:�FW7�       `/�#	�P;�ǩ�At*

epsilon��:��p��-       <A��	 C=�ǩ�Au* 

Average reward per step��:����p       `/�#	�C=�ǩ�Au*

epsilon��:��̴�-       <A��	od?�ǩ�Av* 

Average reward per step��:�R�,�       `/�#	(e?�ǩ�Av*

epsilon��:�ng�-       <A��	�kA�ǩ�Aw* 

Average reward per step��:���#�       `/�#	nlA�ǩ�Aw*

epsilon��:����p-       <A��	|�C�ǩ�Ax* 

Average reward per step��:�5�.       `/�#	�C�ǩ�Ax*

epsilon��:���-       <A��	a�E�ǩ�Ay* 

Average reward per step��:�JC       `/�#	*�E�ǩ�Ay*

epsilon��:�6R�-       <A��	D�G�ǩ�Az* 

Average reward per step��:�H>*]       `/�#	�G�ǩ�Az*

epsilon��:�B΄�-       <A��	�aI�ǩ�A{* 

Average reward per step��:�ȆR�       `/�#	�bI�ǩ�A{*

epsilon��:���;-       <A��	ϽJ�ǩ�A|* 

Average reward per step��:�;E�       `/�#	��J�ǩ�A|*

epsilon��:�_9�Q-       <A��	;�L�ǩ�A}* 

Average reward per step��:�ʤv�       `/�#	��L�ǩ�A}*

epsilon��:���'-       <A��	�N�ǩ�A~* 

Average reward per step��:�z@H�       `/�#	��N�ǩ�A~*

epsilon��:�,�}]-       <A��	(�P�ǩ�A* 

Average reward per step��:��ך�       `/�#	��P�ǩ�A*

epsilon��:�g�~U.       ��W�	h�R�ǩ�A�* 

Average reward per step��:��]>       ��2	1�R�ǩ�A�*

epsilon��:��Fy�.       ��W�	�EU�ǩ�A�* 

Average reward per step��:�����       ��2	|FU�ǩ�A�*

epsilon��:�T��U.       ��W�	��V�ǩ�A�* 

Average reward per step��:�l�k       ��2	;�V�ǩ�A�*

epsilon��:�8 }.       ��W�	S�X�ǩ�A�* 

Average reward per step��:���aT       ��2	!�X�ǩ�A�*

epsilon��:��*vg.       ��W�	��Z�ǩ�A�* 

Average reward per step��:�p~�P       ��2	1�Z�ǩ�A�*

epsilon��:�N�.       ��W�	��\�ǩ�A�* 

Average reward per step��:��y�        ��2	��\�ǩ�A�*

epsilon��:�҆9�.       ��W�	��^�ǩ�A�* 

Average reward per step��:��&A�       ��2	4�^�ǩ�A�*

epsilon��:��(N`.       ��W�	�'`�ǩ�A�* 

Average reward per step��:�rS�w       ��2	�(`�ǩ�A�*

epsilon��:�xI�.       ��W�	�*b�ǩ�A�* 

Average reward per step��:���Bz       ��2	R+b�ǩ�A�*

epsilon��:�&���.       ��W�		2d�ǩ�A�* 

Average reward per step��:��)�       ��2	�2d�ǩ�A�*

epsilon��:����[.       ��W�	�Hf�ǩ�A�* 

Average reward per step��:����&       ��2	�If�ǩ�A�*

epsilon��:��.'�.       ��W�	)wh�ǩ�A�* 

Average reward per step��:�k��       ��2	�wh�ǩ�A�*

epsilon��:�|�.       ��W�	ڪj�ǩ�A�* 

Average reward per step��:���G�       ��2	��j�ǩ�A�*

epsilon��:�#_Z&.       ��W�	f�l�ǩ�A�* 

Average reward per step��:�[�       ��2	4�l�ǩ�A�*

epsilon��:�)��x.       ��W�	f�n�ǩ�A�* 

Average reward per step��:�Ī#       ��2	3�n�ǩ�A�*

epsilon��:���A.       ��W�	��r�ǩ�A�* 

Average reward per step��:����_       ��2	:�r�ǩ�A�*

epsilon��:�V��9.       ��W�	m�t�ǩ�A�* 

Average reward per step��:�a�Zt       ��2	�t�ǩ�A�*

epsilon��:����.       ��W�	�v�ǩ�A�* 

Average reward per step��:�Ȇ��       ��2	��v�ǩ�A�*

epsilon��:��I��.       ��W�	|�x�ǩ�A�* 

Average reward per step��:���       ��2	9�x�ǩ�A�*

epsilon��:��e�W.       ��W�	��z�ǩ�A�* 

Average reward per step��:�]�+]       ��2	��z�ǩ�A�*

epsilon��:��f�.       ��W�	�|�ǩ�A�* 

Average reward per step��:�w)|       ��2	��|�ǩ�A�*

epsilon��:�����.       ��W�	�%�ǩ�A�* 

Average reward per step��:��� �       ��2	�&�ǩ�A�*

epsilon��:�G��.       ��W�	�`��ǩ�A�* 

Average reward per step��:�]yaz       ��2	�a��ǩ�A�*

epsilon��:�N$p.       ��W�	7���ǩ�A�* 

Average reward per step��:���n�       ��2	"��ǩ�A�*

epsilon��:�8U�.       ��W�	����ǩ�A�* 

Average reward per step��:��1       ��2	m���ǩ�A�*

epsilon��:��}.       ��W�	+���ǩ�A�* 

Average reward per step��:��#GF       ��2	���ǩ�A�*

epsilon��:�����.       ��W�	O��ǩ�A�* 

Average reward per step��:��h��       ��2	��ǩ�A�*

epsilon��:���'.       ��W�	����ǩ�A�* 

Average reward per step��:��S!�       ��2	e���ǩ�A�*

epsilon��:��e��.       ��W�	���ǩ�A�* 

Average reward per step��:�A��<       ��2	���ǩ�A�*

epsilon��:��L�.       ��W�	>��ǩ�A�* 

Average reward per step��:�(���       ��2	���ǩ�A�*

epsilon��:��F.       ��W�	���ǩ�A�* 

Average reward per step��:��~2~       ��2	���ǩ�A�*

epsilon��:��j�.       ��W�	����ǩ�A�* 

Average reward per step��:��P\�       ��2	v���ǩ�A�*

epsilon��:�z���.       ��W�		��ǩ�A�* 

Average reward per step��:�sB�-       ��2	�	��ǩ�A�*

epsilon��:���ݹ.       ��W�	���ǩ�A�* 

Average reward per step��:�u��       ��2	]��ǩ�A�*

epsilon��:�]��.       ��W�	(��ǩ�A�* 

Average reward per step��:�WZ��       ��2	�(��ǩ�A�*

epsilon��:����.       ��W�	���ǩ�A�* 

Average reward per step��:�	��       ��2	͐��ǩ�A�*

epsilon��:�/)G�.       ��W�	�ǝ�ǩ�A�* 

Average reward per step��:���{�       ��2	�ȝ�ǩ�A�*

epsilon��:�b��0       ���_	K��ǩ�A*#
!
Average reward per episodes�@�U@2.       ��W�	���ǩ�A*!

total reward per episode  P��,�.       ��W�	���ǩ�A�* 

Average reward per steps�@�-�`B       ��2	����ǩ�A�*

epsilons�@�q�7�.       ��W�	���ǩ�A�* 

Average reward per steps�@���!       ��2	t��ǩ�A�*

epsilons�@��k��.       ��W�	���ǩ�A�* 

Average reward per steps�@��6��       ��2	㍤�ǩ�A�*

epsilons�@�g�:�.       ��W�	Է��ǩ�A�* 

Average reward per steps�@����       ��2	o���ǩ�A�*

epsilons�@�G���.       ��W�	�w��ǩ�A�* 

Average reward per steps�@�>��>       ��2	�x��ǩ�A�*

epsilons�@��Z�/.       ��W�	���ǩ�A�* 

Average reward per steps�@����       ��2	��ǩ�A�*

epsilons�@���".       ��W�	4ج�ǩ�A�* 

Average reward per steps�@�����       ��2	٬�ǩ�A�*

epsilons�@�t��.       ��W�	�ڮ�ǩ�A�* 

Average reward per steps�@��R��       ��2	@ۮ�ǩ�A�*

epsilons�@�Mܫ�.       ��W�	$��ǩ�A�* 

Average reward per steps�@�`=_       ��2	=��ǩ�A�*

epsilons�@����.       ��W�	���ǩ�A�* 

Average reward per steps�@��tH       ��2	z��ǩ�A�*

epsilons�@��Te�.       ��W�	���ǩ�A�* 

Average reward per steps�@��p�       ��2	���ǩ�A�*

epsilons�@��oo�.       ��W�	[B��ǩ�A�* 

Average reward per steps�@�OՍ"       ��2	JC��ǩ�A�*

epsilons�@���O�.       ��W�	�W��ǩ�A�* 

Average reward per steps�@��%�       ��2	GX��ǩ�A�*

epsilons�@��^�.       ��W�	�l��ǩ�A�* 

Average reward per steps�@���A�       ��2	jm��ǩ�A�*

epsilons�@���z�.       ��W�	x��ǩ�A�* 

Average reward per steps�@��W1x       ��2	�x��ǩ�A�*

epsilons�@��N�.       ��W�	�p��ǩ�A�* 

Average reward per steps�@����       ��2	zq��ǩ�A�*

epsilons�@�i���.       ��W�	o~��ǩ�A�* 

Average reward per steps�@��n�4       ��2	9��ǩ�A�*

epsilons�@�$r\q.       ��W�	����ǩ�A�* 

Average reward per steps�@�����       ��2	����ǩ�A�*

epsilons�@�W/5�.       ��W�	%���ǩ�A�* 

Average reward per steps�@���$�       ��2	���ǩ�A�*

epsilons�@��;�.       ��W�	8���ǩ�A�* 

Average reward per steps�@�o�/[       ��2	
���ǩ�A�*

epsilons�@�n3�0.       ��W�	З��ǩ�A�* 

Average reward per steps�@�#g�s       ��2	����ǩ�A�*

epsilons�@��<��.       ��W�	����ǩ�A�* 

Average reward per steps�@�v[�       ��2	Y���ǩ�A�*

epsilons�@��%�.       ��W�	~���ǩ�A�* 

Average reward per steps�@�%P�       ��2	6���ǩ�A�*

epsilons�@�&`�.       ��W�	����ǩ�A�* 

Average reward per steps�@�����       ��2	���ǩ�A�*

epsilons�@��t��.       ��W�	����ǩ�A�* 

Average reward per steps�@�-'1�       ��2	S���ǩ�A�*

epsilons�@����.       ��W�	����ǩ�A�* 

Average reward per steps�@����       ��2	����ǩ�A�*

epsilons�@���d.       ��W�	\��ǩ�A�* 

Average reward per steps�@�/���       ��2	��ǩ�A�*

epsilons�@�k�T�.       ��W�	%��ǩ�A�* 

Average reward per steps�@�$��U       ��2	�%��ǩ�A�*

epsilons�@��.ި.       ��W�	m8��ǩ�A�* 

Average reward per steps�@���       ��2	9��ǩ�A�*

epsilons�@�����.       ��W�	`9��ǩ�A�* 

Average reward per steps�@�^B2�       ��2	T:��ǩ�A�*

epsilons�@���2.       ��W�	MM��ǩ�A�* 

Average reward per steps�@�5Y�	       ��2	jN��ǩ�A�*

epsilons�@�|��6.       ��W�	����ǩ�A�* 

Average reward per steps�@�\��f       ��2	����ǩ�A�*

epsilons�@�����.       ��W�	����ǩ�A�* 

Average reward per steps�@�>��       ��2	t���ǩ�A�*

epsilons�@��u�{.       ��W�	Uh��ǩ�A�* 

Average reward per steps�@��9�       ��2	i��ǩ�A�*

epsilons�@����h.       ��W�	����ǩ�A�* 

Average reward per steps�@���       ��2	����ǩ�A�*

epsilons�@���T.       ��W�	���ǩ�A�* 

Average reward per steps�@��7S       ��2	����ǩ�A�*

epsilons�@�4�[;.       ��W�	�z��ǩ�A�* 

Average reward per steps�@�       ��2	�{��ǩ�A�*

epsilons�@�ڦ2�.       ��W�	���ǩ�A�* 

Average reward per steps�@�%ʘ�       ��2	?���ǩ�A�*

epsilons�@�Y�!.       ��W�		���ǩ�A�* 

Average reward per steps�@��n       ��2	����ǩ�A�*

epsilons�@�[�&�.       ��W�	�2��ǩ�A�* 

Average reward per steps�@�d{5�       ��2	Y3��ǩ�A�*

epsilons�@���H 0       ���_	�P��ǩ�A*#
!
Average reward per episode��4�)�+�.       ��W�	DQ��ǩ�A*!

total reward per episode  ���WGo.       ��W�	�2��ǩ�A�* 

Average reward per step��4���H	       ��2	Q3��ǩ�A�*

epsilon��4���kQ.       ��W�	_���ǩ�A�* 

Average reward per step��4�o�Y�       ��2	ٗ��ǩ�A�*

epsilon��4�rɪ.       ��W�	&���ǩ�A�* 

Average reward per step��4�z�Dw       ��2	Ҭ��ǩ�A�*

epsilon��4�7N�.       ��W�	���ǩ�A�* 

Average reward per step��4�Z{       ��2	����ǩ�A�*

epsilon��4���_.       ��W�	����ǩ�A�* 

Average reward per step��4��4+�       ��2	����ǩ�A�*

epsilon��4��fQ.       ��W�	�0�ǩ�A�* 

Average reward per step��4�C��       ��2	�1�ǩ�A�*

epsilon��4�K��.       ��W�	Uj�ǩ�A�* 

Average reward per step��4�7��C       ��2	�j�ǩ�A�*

epsilon��4��C<.       ��W�	���ǩ�A�* 

Average reward per step��4�<��       ��2	q��ǩ�A�*

epsilon��4��@�.       ��W�	F��ǩ�A�* 

Average reward per step��4�^:/�       ��2	���ǩ�A�*

epsilon��4�Ѓެ.       ��W�	�d	�ǩ�A�* 

Average reward per step��4�`K<       ��2	Ze	�ǩ�A�*

epsilon��4���mP.       ��W�	-�ǩ�A�* 

Average reward per step��4��9^       ��2	��ǩ�A�*

epsilon��4�s�*.       ��W�	�ǩ�A�* 

Average reward per step��4��!�       ��2	U�ǩ�A�*

epsilon��4� ���.       ��W�	��ǩ�A�* 

Average reward per step��4�izo       ��2	��ǩ�A�*

epsilon��4��_�Z.       ��W�	�&�ǩ�A�* 

Average reward per step��4���I[       ��2	�'�ǩ�A�*

epsilon��4�O���0       ���_	NA�ǩ�A*#
!
Average reward per episode  H����.       ��W�	�A�ǩ�A*!

total reward per episode  /ïŎ�.       ��W�	�U�ǩ�A�* 

Average reward per step  H���<       ��2	�V�ǩ�A�*

epsilon  H��hm�.       ��W�	i��ǩ�A�* 

Average reward per step  H��k       ��2	��ǩ�A�*

epsilon  H�'�8�.       ��W�	�O�ǩ�A�* 

Average reward per step  H�e���       ��2	�P�ǩ�A�*

epsilon  H����.       ��W�	ߦ�ǩ�A�* 

Average reward per step  H���[       ��2	ۧ�ǩ�A�*

epsilon  H���7.       ��W�	���ǩ�A�* 

Average reward per step  H�
��.       ��2	���ǩ�A�*

epsilon  H�A��	.       ��W�	~Q�ǩ�A�* 

Average reward per step  H��0�       ��2	]R�ǩ�A�*

epsilon  H����.       ��W�	GU!�ǩ�A�* 

Average reward per step  H��JL�       ��2	V!�ǩ�A�*

epsilon  H��L�t.       ��W�	i#�ǩ�A�* 

Average reward per step  H��\�f       ��2	�i#�ǩ�A�*

epsilon  H�����.       ��W�	�`%�ǩ�A�* 

Average reward per step  H���S�       ��2	[a%�ǩ�A�*

epsilon  H����.       ��W�	t'�ǩ�A�* 

Average reward per step  H���tZ       ��2	�t'�ǩ�A�*

epsilon  H��m$.       ��W�	�w)�ǩ�A�* 

Average reward per step  H���@y       ��2	�x)�ǩ�A�*

epsilon  H�<
�.       ��W�	�+�ǩ�A�* 

Average reward per step  H��x       ��2	@�+�ǩ�A�*

epsilon  H��Df.       ��W�	k�-�ǩ�A�* 

Average reward per step  H��sH       ��2	�-�ǩ�A�*

epsilon  H�K��.       ��W�	��/�ǩ�A�* 

Average reward per step  H�����       ��2	(�/�ǩ�A�*

epsilon  H��m�a.       ��W�	��1�ǩ�A�* 

Average reward per step  H����N       ��2	��1�ǩ�A�*

epsilon  H��h�.       ��W�	A3�ǩ�A�* 

Average reward per step  H�3�N�       ��2	�3�ǩ�A�*

epsilon  H�����.       ��W�	�5�ǩ�A�* 

Average reward per step  H��9QA       ��2	��5�ǩ�A�*

epsilon  H����.       ��W�	�7�ǩ�A�* 

Average reward per step  H��L��       ��2	}�7�ǩ�A�*

epsilon  H����0       ���_	J�7�ǩ�A*#
!
Average reward per episode�	�]�e�.       ��W�	��7�ǩ�A*!

total reward per episode  ���p.       ��W�		<�ǩ�A�* 

Average reward per step�	����;       ��2	�	<�ǩ�A�*

epsilon�	����.       ��W�	v4>�ǩ�A�* 

Average reward per step�	��9u�       ��2	H5>�ǩ�A�*

epsilon�	�y3�.       ��W�	?W@�ǩ�A�* 

Average reward per step�	����       ��2	CX@�ǩ�A�*

epsilon�	��r%|.       ��W�	��A�ǩ�A�* 

Average reward per step�	�\��       ��2	q�A�ǩ�A�*

epsilon�	����.       ��W�	}�D�ǩ�A�* 

Average reward per step�	����       ��2	`�D�ǩ�A�*

epsilon�	�Y^�.       ��W�	�5H�ǩ�A�* 

Average reward per step�	���#1       ��2	�6H�ǩ�A�*

epsilon�	��?�h.       ��W�	]�J�ǩ�A�* 

Average reward per step�	���B       ��2	Q�J�ǩ�A�*

epsilon�	�5��l.       ��W�	�8L�ǩ�A�* 

Average reward per step�	��C�       ��2	�9L�ǩ�A�*

epsilon�	�¼M�.       ��W�	^O�ǩ�A�* 

Average reward per step�	�j�l       ��2	_O�ǩ�A�*

epsilon�	��1�.       ��W�	��R�ǩ�A�* 

Average reward per step�	�{��       ��2	8�R�ǩ�A�*

epsilon�	�%ou�.       ��W�	��T�ǩ�A�* 

Average reward per step�	�e`�c       ��2	n�T�ǩ�A�*

epsilon�	�c��i.       ��W�	�vV�ǩ�A�* 

Average reward per step�	��8#       ��2	�wV�ǩ�A�*

epsilon�	���8\.       ��W�	��X�ǩ�A�* 

Average reward per step�	�!�/6       ��2	��X�ǩ�A�*

epsilon�	��Bt^.       ��W�	g�Z�ǩ�A�* 

Average reward per step�	��>	�       ��2	�Z�ǩ�A�*

epsilon�	��7�.       ��W�	��\�ǩ�A�* 

Average reward per step�	�W��Z       ��2	g�\�ǩ�A�*

epsilon�	�V�f�.       ��W�		�^�ǩ�A�* 

Average reward per step�	��-�5       ��2	��^�ǩ�A�*

epsilon�	��Qv]0       ���_	�_�ǩ�A*#
!
Average reward per episode  -�z\�.       ��W�	@_�ǩ�A*!

total reward per episode  -�0���.       ��W�	o�b�ǩ�A�* 

Average reward per step  -��Fg>       ��2	�b�ǩ�A�*

epsilon  -�:��f.       ��W�	� e�ǩ�A�* 

Average reward per step  -��R       ��2	�!e�ǩ�A�*

epsilon  -����.       ��W�	�f�ǩ�A�* 

Average reward per step  -��S��       ��2	ٗf�ǩ�A�*

epsilon  -�_�M.       ��W�	�fi�ǩ�A�* 

Average reward per step  -�x�!�       ��2	�gi�ǩ�A�*

epsilon  -�Ƚ�.       ��W�	��j�ǩ�A�* 

Average reward per step  -�}U��       ��2	/�j�ǩ�A�*

epsilon  -��̕�.       ��W�	�
m�ǩ�A�* 

Average reward per step  -�&�Cf       ��2	�m�ǩ�A�*

epsilon  -�9��k.       ��W�	�ro�ǩ�A�* 

Average reward per step  -���t       ��2	�so�ǩ�A�*

epsilon  -�����.       ��W�	��p�ǩ�A�* 

Average reward per step  -�BSf�       ��2	]�p�ǩ�A�*

epsilon  -��O�..       ��W�	F�s�ǩ�A�* 

Average reward per step  -����       ��2	)�s�ǩ�A�*

epsilon  -�A�1.       ��W�	{u�ǩ�A�* 

Average reward per step  -�	�       ��2	u�ǩ�A�*

epsilon  -�,2+.       ��W�	�w�ǩ�A�* 

Average reward per step  -��pt       ��2	*w�ǩ�A�*

epsilon  -���:�.       ��W�	Ny�ǩ�A�* 

Average reward per step  -���r�       ��2	Oy�ǩ�A�*

epsilon  -�G��.       ��W�	%�{�ǩ�A�* 

Average reward per step  -�ɐ       ��2	�{�ǩ�A�*

epsilon  -�n�ۥ.       ��W�	(�}�ǩ�A�* 

Average reward per step  -�M !�       ��2	��}�ǩ�A�*

epsilon  -���{.       ��W�	�n�ǩ�A�* 

Average reward per step  -����       ��2	�o�ǩ�A�*

epsilon  -�Md�f.       ��W�	m��ǩ�A�* 

Average reward per step  -�L�<�       ��2	�m��ǩ�A�*

epsilon  -�G�C1.       ��W�	�x��ǩ�A�* 

Average reward per step  -��	)�       ��2	�y��ǩ�A�*

epsilon  -�M"t.       ��W�	�^��ǩ�A�* 

Average reward per step  -��Y�       ��2	 _��ǩ�A�*

epsilon  -���e1.       ��W�	+P��ǩ�A�* 

Average reward per step  -��"x1       ��2	�P��ǩ�A�*

epsilon  -�P9�0       ���_	Uk��ǩ�A*#
!
Average reward per episode�k�LA2:.       ��W�	�k��ǩ�A*!

total reward per episode  "Ö�T�.       ��W�	�@��ǩ�A�* 

Average reward per step�k�?2       ��2	SA��ǩ�A�*

epsilon�k��B.       ��W�	N���ǩ�A�* 

Average reward per step�k�4*        ��2	Ԁ��ǩ�A�*

epsilon�k����.       ��W�	5z��ǩ�A�* 

Average reward per step�k�iB��       ��2	{��ǩ�A�*

epsilon�k�\/�.       ��W�	�x��ǩ�A�* 

Average reward per step�k��Ϙ�       ��2	hy��ǩ�A�*

epsilon�k��J��.       ��W�	>Z��ǩ�A�* 

Average reward per step�k��KS�       ��2	[��ǩ�A�*

epsilon�k�ݦ��.       ��W�	?���ǩ�A�* 

Average reward per step�k�s�O-       ��2	���ǩ�A�*

epsilon�k�n9�.       ��W�	����ǩ�A�* 

Average reward per step�k��%?8       ��2	o���ǩ�A�*

epsilon�k��Q6�.       ��W�	���ǩ�A�* 

Average reward per step�k��8��       ��2	����ǩ�A�*

epsilon�k�6��.       ��W�	��ǩ�A�* 

Average reward per step�k��       ��2	����ǩ�A�*

epsilon�k����.       ��W�	�_��ǩ�A�* 

Average reward per step�k�X=�       ��2	�`��ǩ�A�*

epsilon�k�]].       ��W�	Xɝ�ǩ�A�* 

Average reward per step�k�����       ��2	�ɝ�ǩ�A�*

epsilon�k���X.       ��W�	�J��ǩ�A�* 

Average reward per step�k���=G       ��2	L��ǩ�A�*

epsilon�k���U.       ��W�	����ǩ�A�* 

Average reward per step�k�W3ذ       ��2	o���ǩ�A�*

epsilon�k���.       ��W�	u��ǩ�A�* 

Average reward per step�k��G.       ��2	6 ��ǩ�A�*

epsilon�k���B.       ��W�	�)��ǩ�A�* 

Average reward per step�k�G��V       ��2	�*��ǩ�A�*

epsilon�k��.E.       ��W�	b���ǩ�A�* 

Average reward per step�k���       ��2	���ǩ�A�*

epsilon�k��rV�.       ��W�	�u��ǩ�A�* 

Average reward per step�k�Xݽ|       ��2	6v��ǩ�A�*

epsilon�k��	.       ��W�	�b��ǩ�A�* 

Average reward per step�k�I=�W       ��2	�c��ǩ�A�*

epsilon�k���5.       ��W�	.T��ǩ�A�* 

Average reward per step�k��$       ��2	�T��ǩ�A�*

epsilon�k�Uu�.       ��W�	����ǩ�A�* 

Average reward per step�k��]|�       ��2	����ǩ�A�*

epsilon�k����+.       ��W�	Ct��ǩ�A�* 

Average reward per step�k���       ��2	u��ǩ�A�*

epsilon�k��'�.       ��W�	�q��ǩ�A�* 

Average reward per step�k����$       ��2	�r��ǩ�A�*

epsilon�k�oz�u.       ��W�	�b��ǩ�A�* 

Average reward per step�k���S<       ��2	^c��ǩ�A�*

epsilon�k�GZ�.       ��W�	Q���ǩ�A�* 

Average reward per step�k���~x       ��2	#���ǩ�A�*

epsilon�k�ǋa,0       ���_	�ֶ�ǩ�A*#
!
Average reward per episode  ����X.       ��W�	�׶�ǩ�A*!

total reward per episode  �S�h.       ��W�	���ǩ�A�* 

Average reward per step  ��KG��       ��2	}��ǩ�A�*

epsilon  ����l.       ��W�	���ǩ�A�* 

Average reward per step  ��'Ie�       ��2	���ǩ�A�*

epsilon  ��͙tC.       ��W�	R��ǩ�A�* 

Average reward per step  ��ZW�       ��2	���ǩ�A�*

epsilon  ���#�.       ��W�	W���ǩ�A�* 

Average reward per step  ��k���       ��2	$���ǩ�A�*

epsilon  �����.       ��W�	����ǩ�A�* 

Average reward per step  ��}�Q�       ��2	~���ǩ�A�*

epsilon  ��rb�.       ��W�	*���ǩ�A�* 

Average reward per step  ��5j�       ��2	����ǩ�A�*

epsilon  ��_ V.       ��W�	����ǩ�A�* 

Average reward per step  �����       ��2	X���ǩ�A�*

epsilon  ��6sf�.       ��W�	���ǩ�A�* 

Average reward per step  �����       ��2	���ǩ�A�*

epsilon  ��p�c.       ��W�	_��ǩ�A�* 

Average reward per step  ��&,'8       ��2	���ǩ�A�*

epsilon  ��a��.       ��W�	,��ǩ�A�* 

Average reward per step  ���.V�       ��2	���ǩ�A�*

epsilon  ���G�.       ��W�	���ǩ�A�* 

Average reward per step  ������       ��2	���ǩ�A�*

epsilon  ��r8A�0       ���_	&4��ǩ�A	*#
!
Average reward per episodeF}�VM��.       ��W�	�4��ǩ�A	*!

total reward per episode  .þ��.       ��W�	.��ǩ�A�* 

Average reward per stepF}��{       ��2	�.��ǩ�A�*

epsilonF}��bf.       ��W�	�+��ǩ�A�* 

Average reward per stepF}�!�@       ��2	Z,��ǩ�A�*

epsilonF}�A��.       ��W�	���ǩ�A�* 

Average reward per stepF}�#(X       ��2	��ǩ�A�*

epsilonF}�1�J�.       ��W�	 '��ǩ�A�* 

Average reward per stepF}�gg�       ��2	(��ǩ�A�*

epsilonF}���e�.       ��W�	�C��ǩ�A�* 

Average reward per stepF}���>       ��2	cD��ǩ�A�*

epsilonF}���p.       ��W�	b.��ǩ�A�* 

Average reward per stepF}��5�       ��2	�.��ǩ�A�*

epsilonF}�dI�a.       ��W�	�o��ǩ�A�* 

Average reward per stepF}����       ��2	�p��ǩ�A�*

epsilonF}�� !�.       ��W�	7o��ǩ�A�* 

Average reward per stepF}�6�Q%       ��2	p��ǩ�A�*

epsilonF}�Z>v.       ��W�	g���ǩ�A�* 

Average reward per stepF}��԰x       ��2	����ǩ�A�*

epsilonF}�-��.       ��W�	˝��ǩ�A�* 

Average reward per stepF}�$��T       ��2	^���ǩ�A�*

epsilonF}��_��.       ��W�	l���ǩ�A�* 

Average reward per stepF}�43b       ��2	����ǩ�A�*

epsilonF}���_P.       ��W�	����ǩ�A�* 

Average reward per stepF}�L2Gi       ��2	����ǩ�A�*

epsilonF}��g82.       ��W�	����ǩ�A�* 

Average reward per stepF}��`�U       ��2	M���ǩ�A�*

epsilonF}�i!�,.       ��W�	_���ǩ�A�* 

Average reward per stepF}�I�}o       ��2	W���ǩ�A�*

epsilonF}�]�=�.       ��W�	����ǩ�A�* 

Average reward per stepF}���`?       ��2	���ǩ�A�*

epsilonF}�EG/}.       ��W�	Ae��ǩ�A�* 

Average reward per stepF}�S"�       ��2	f��ǩ�A�*

epsilonF}�+}�.       ��W�	�5��ǩ�A�* 

Average reward per stepF}���Ҫ       ��2	�6��ǩ�A�*

epsilonF}�l2J.       ��W�	+1��ǩ�A�* 

Average reward per stepF}�����       ��2	�1��ǩ�A�*

epsilonF}�ETX.       ��W�	����ǩ�A�* 

Average reward per stepF}�;��       ��2	`���ǩ�A�*

epsilonF}�$wW�.       ��W�	z���ǩ�A�* 

Average reward per stepF}��"�H       ��2	 ��ǩ�A�*

epsilonF}��S2W.       ��W�	����ǩ�A�* 

Average reward per stepF}��&)       ��2	~���ǩ�A�*

epsilonF}��#>�.       ��W�	����ǩ�A�* 

Average reward per stepF}�|
3j       ��2	� ��ǩ�A�*

epsilonF}���E�.       ��W�	A��ǩ�A�* 

Average reward per stepF}���ч       ��2	(��ǩ�A�*

epsilonF}�#$xp.       ��W�	��ǩ�A�* 

Average reward per stepF}�w5P�       ��2	��ǩ�A�*

epsilonF}�#�O�0       ���_	>&�ǩ�A
*#
!
Average reward per episode����!��5.       ��W�	�&�ǩ�A
*!

total reward per episode  �3Nʀ.       ��W�	�%�ǩ�A�* 

Average reward per step����*�q�       ��2	�&�ǩ�A�*

epsilon����0�g.       ��W�	a�ǩ�A�* 

Average reward per step����L�S       ��2	�ǩ�A�*

epsilon�����1�.       ��W�	kd�ǩ�A�* 

Average reward per step����z�A       ��2	e�ǩ�A�*

epsilon������:�.       ��W�	d\
�ǩ�A�* 

Average reward per step����gP$       ��2	�\
�ǩ�A�*

epsilon����A�a�.       ��W�	�G�ǩ�A�* 

Average reward per step��������       ��2	�H�ǩ�A�*

epsilon����5�=.       ��W�	�>�ǩ�A�* 

Average reward per step�����>       ��2	?�ǩ�A�*

epsilon����)�U�.       ��W�	���ǩ�A�* 

Average reward per step����^m�       ��2	z��ǩ�A�*

epsilon����i�.       ��W�	�{�ǩ�A�* 

Average reward per step����A9�<       ��2	�|�ǩ�A�*

epsilon����؃��.       ��W�	<l�ǩ�A�* 

Average reward per step������o�       ��2	�l�ǩ�A�*

epsilon�����DK�.       ��W�	�b�ǩ�A�* 

Average reward per step����|�p       ��2	�c�ǩ�A�*

epsilon����ViH.       ��W�	�e�ǩ�A�* 

Average reward per step��������       ��2	�f�ǩ�A�*

epsilon�������.       ��W�	!��ǩ�A�* 

Average reward per step����N�1       ��2	��ǩ�A�*

epsilon����pf��.       ��W�	ę�ǩ�A�* 

Average reward per step�����KQq       ��2	N��ǩ�A�*

epsilon��������.       ��W�	��ǩ�A�* 

Average reward per step�������       ��2	��ǩ�A�*

epsilon����0���.       ��W�	-��ǩ�A�* 

Average reward per step�����R|g       ��2	���ǩ�A�*

epsilon����i_U�.       ��W�	ƈ �ǩ�A�* 

Average reward per step�����w�       ��2	U� �ǩ�A�*

epsilon����=s�	.       ��W�	a�"�ǩ�A�* 

Average reward per step����"�Y�       ��2	�"�ǩ�A�*

epsilon������'.       ��W�	�$�ǩ�A�* 

Average reward per step����9��       ��2	��$�ǩ�A�*

epsilon����5}1�.       ��W�	��&�ǩ�A�* 

Average reward per step����� =       ��2	ő&�ǩ�A�*

epsilon����y$��0       ���_	N�&�ǩ�A*#
!
Average reward per episodey�$J0�.       ��W�	�&�ǩ�A*!

total reward per episode  ����.       ��W�	}�*�ǩ�A�* 

Average reward per stepy����       ��2	��*�ǩ�A�*

epsilony���j .       ��W�	"�,�ǩ�A�* 

Average reward per stepy�^0w       ��2	��,�ǩ�A�*

epsilony�9��.       ��W�	��.�ǩ�A�* 

Average reward per stepy����4       ��2	'�.�ǩ�A�*

epsilony��T�.       ��W�	��0�ǩ�A�* 

Average reward per stepy���!f       ��2	��0�ǩ�A�*

epsilony�9��5.       ��W�	��2�ǩ�A�* 

Average reward per stepy�"$��       ��2	�2�ǩ�A�*

epsilony�@'�.       ��W�	��4�ǩ�A�* 

Average reward per stepy�]g[g       ��2	��4�ǩ�A�*

epsilony��k�.       ��W�	�6�ǩ�A�* 

Average reward per stepy�;�ц       ��2	��6�ǩ�A�*

epsilony�zZu�.       ��W�	�+8�ǩ�A�* 

Average reward per stepy�)s�       ��2	�,8�ǩ�A�*

epsilony�IG��.       ��W�	-:�ǩ�A�* 

Average reward per stepy�b�*�       ��2	�-:�ǩ�A�*

epsilony�]�.       ��W�	�K<�ǩ�A�* 

Average reward per stepy�)��`       ��2	�L<�ǩ�A�*

epsilony�퓠�.       ��W�	rS>�ǩ�A�* 

Average reward per stepy�5��       ��2	T>�ǩ�A�*

epsilony���.       ��W�	�a@�ǩ�A�* 

Average reward per stepy�����       ��2	�b@�ǩ�A�*

epsilony�S�α.       ��W�	�xB�ǩ�A�* 

Average reward per stepy��
d       ��2	�yB�ǩ�A�*

epsilony�X_�.       ��W�	��D�ǩ�A�* 

Average reward per stepy�n�j       ��2	�D�ǩ�A�*

epsilony�Q�j.       ��W�	��F�ǩ�A�* 

Average reward per stepy�'��       ��2	��F�ǩ�A�*

epsilony�<�@.       ��W�	V�H�ǩ�A�* 

Average reward per stepy��tѾ       ��2	A�H�ǩ�A�*

epsilony�wT*�.       ��W�	�J�ǩ�A�* 

Average reward per stepy�IM�       ��2	�J�ǩ�A�*

epsilony��{��.       ��W�	�~K�ǩ�A�* 

Average reward per stepy��ά�       ��2	K�ǩ�A�*

epsilony��5I�.       ��W�	�M�ǩ�A�* 

Average reward per stepy��"S�       ��2	�M�ǩ�A�*

epsilony�?iN(.       ��W�	��O�ǩ�A�* 

Average reward per stepy��^�H       ��2	r�O�ǩ�A�*

epsilony���<�.       ��W�	v�Q�ǩ�A�* 

Average reward per stepy�w�2       ��2	�Q�ǩ�A�*

epsilony��Ԣ�.       ��W�	��S�ǩ�A�* 

Average reward per stepy�:+U       ��2	��S�ǩ�A�*

epsilony� �P.       ��W�	�U�ǩ�A�* 

Average reward per stepy��C�       ��2	��U�ǩ�A�*

epsilony��L�.       ��W�	�W�ǩ�A�* 

Average reward per stepy��       ��2	g�W�ǩ�A�*

epsilony�V��O.       ��W�	��Y�ǩ�A�* 

Average reward per stepy�@��       ��2	��Y�ǩ�A�*

epsilony�1�\�.       ��W�	��[�ǩ�A�* 

Average reward per stepy���S       ��2	2�[�ǩ�A�*

epsilony��/.       ��W�	��]�ǩ�A�* 

Average reward per stepy�z.M�       ��2	��]�ǩ�A�*

epsilony�I)�H.       ��W�	}�_�ǩ�A�* 

Average reward per stepy���]�       ��2	�_�ǩ�A�*

epsilony��~�.       ��W�	��a�ǩ�A�* 

Average reward per stepy�w�H       ��2	��a�ǩ�A�*

epsilony�=�.       ��W�	��c�ǩ�A�* 

Average reward per stepy����       ��2	��c�ǩ�A�*

epsilony�����.       ��W�	�f�ǩ�A�* 

Average reward per stepy��wA�       ��2	�f�ǩ�A�*

epsilony��B��.       ��W�	�h�ǩ�A�* 

Average reward per stepy�����       ��2	mh�ǩ�A�*

epsilony�27̱.       ��W�	&j�ǩ�A�* 

Average reward per stepy��N<�       ��2	�&j�ǩ�A�*

epsilony�g�)�.       ��W�	N+l�ǩ�A�* 

Average reward per stepy�:w�Y       ��2	8,l�ǩ�A�*

epsilony�|��!0       ���_	#Ml�ǩ�A*#
!
Average reward per episode  `�q�c.       ��W�	�Ml�ǩ�A*!

total reward per episode  ��.       ��W�	�	o�ǩ�A�* 

Average reward per step  `���-�       ��2	�
o�ǩ�A�*

epsilon  `��+�g.       ��W�	(q�ǩ�A�* 

Average reward per step  `��l>&       ��2	�q�ǩ�A�*

epsilon  `�;j�.       ��W�	�s�ǩ�A�* 

Average reward per step  `���8       ��2	�s�ǩ�A�*

epsilon  `��]<;.       ��W�	�'u�ǩ�A�* 

Average reward per step  `�Kǹ�       ��2	R(u�ǩ�A�*

epsilon  `� �Mz.       ��W�	#w�ǩ�A�* 

Average reward per step  `��4Ah       ��2	�#w�ǩ�A�*

epsilon  `���=	.       ��W�	�/y�ǩ�A�* 

Average reward per step  `�^g�       ��2	@0y�ǩ�A�*

epsilon  `��mj�.       ��W�	�7{�ǩ�A�* 

Average reward per step  `���k$       ��2	�8{�ǩ�A�*

epsilon  `�	���.       ��W�	MN}�ǩ�A�* 

Average reward per step  `�X��       ��2	�N}�ǩ�A�*

epsilon  `�H���.       ��W�	sG�ǩ�A�* 

Average reward per step  `��+��       ��2	H�ǩ�A�*

epsilon  `���è.       ��W�	�Q��ǩ�A�* 

Average reward per step  `�$       ��2	*R��ǩ�A�*

epsilon  `�c�<.       ��W�	�Z��ǩ�A�* 

Average reward per step  `��g:N       ��2	p[��ǩ�A�*

epsilon  `���E.       ��W�	�h��ǩ�A�* 

Average reward per step  `��"i�       ��2	Ui��ǩ�A�*

epsilon  `���b�.       ��W�	�p��ǩ�A�* 

Average reward per step  `�z��       ��2	mq��ǩ�A�*

epsilon  `��!.       ��W�	����ǩ�A�* 

Average reward per step  `��z�       ��2	]���ǩ�A�*

epsilon  `�&�7.       ��W�	8f��ǩ�A�* 

Average reward per step  `��Q�       ��2	�f��ǩ�A�*

epsilon  `�q��.       ��W�	s��ǩ�A�* 

Average reward per step  `��D!       ��2	�s��ǩ�A�*

epsilon  `�f��.       ��W�	o���ǩ�A�* 

Average reward per step  `�>zO~       ��2	b���ǩ�A�*

epsilon  `���P .       ��W�	S���ǩ�A�* 

Average reward per step  `��!�]       ��2	�ǩ�A�*

epsilon  `�1.       ��W�	l���ǩ�A�* 

Average reward per step  `�&��       ��2	_���ǩ�A�*

epsilon  `�ZI�.       ��W�	Lĕ�ǩ�A�* 

Average reward per step  `��6�)       ��2	&ŕ�ǩ�A�*

epsilon  `��9F�.       ��W�		���ǩ�A�* 

Average reward per step  `��0�g       ��2	����ǩ�A�*

epsilon  `����.       ��W�	���ǩ�A�* 

Average reward per step  `�}푮       ��2	d��ǩ�A�*

epsilon  `�
]jT.       ��W�	P���ǩ�A�* 

Average reward per step  `��<h�       ��2	���ǩ�A�*

epsilon  `���~2.       ��W�	���ǩ�A�* 

Average reward per step  `���u       ��2	A��ǩ�A�*

epsilon  `��}��.       ��W�	�	��ǩ�A�* 

Average reward per step  `��       ��2	�
��ǩ�A�*

epsilon  `�k�!.       ��W�	���ǩ�A�* 

Average reward per step  `��Fg       ��2	x��ǩ�A�*

epsilon  `�f�|^.       ��W�	:��ǩ�A�* 

Average reward per step  `�tcxE       ��2	:<��ǩ�A�*

epsilon  `�����.       ��W�	�P��ǩ�A�* 

Average reward per step  `��r�       ��2	�Q��ǩ�A�*

epsilon  `����_.       ��W�	�C��ǩ�A�* 

Average reward per step  `���       ��2	�D��ǩ�A�*

epsilon  `���$�.       ��W�	L��ǩ�A�* 

Average reward per step  `�s�       ��2	�L��ǩ�A�*

epsilon  `�j� '0       ���_	ٔ��ǩ�A*#
!
Average reward per episode��}��7�Q.       ��W�	h���ǩ�A*!

total reward per episode  �µ�V.       ��W�	s���ǩ�A�* 

Average reward per step��}�NF       ��2	<���ǩ�A�*

epsilon��}���.       ��W�	$	��ǩ�A�* 

Average reward per step��}�.��\       ��2	
��ǩ�A�*

epsilon��}���6�.       ��W�	���ǩ�A�* 

Average reward per step��}�̎�       ��2	6��ǩ�A�*

epsilon��}�`�E�.       ��W�	��ǩ�A�* 

Average reward per step��}�{�Sp       ��2	���ǩ�A�*

epsilon��}�j_��.       ��W�	z��ǩ�A�* 

Average reward per step��}��C�       ��2	L��ǩ�A�*

epsilon��}��,�.       ��W�	�$��ǩ�A�* 

Average reward per step��}��A��       ��2	&��ǩ�A�*

epsilon��}��J�.       ��W�	�3��ǩ�A�* 

Average reward per step��}��Ƭ       ��2	�4��ǩ�A�*

epsilon��}��|1�.       ��W�	+3��ǩ�A�* 

Average reward per step��}��<�       ��2	4��ǩ�A�*

epsilon��}��v.       ��W�	�=��ǩ�A�* 

Average reward per step��}�.�|       ��2	�>��ǩ�A�*

epsilon��}�d��&.       ��W�	��ǩ�A�* 

Average reward per step��}�	�K�       ��2	՗��ǩ�A�*

epsilon��}���s.       ��W�	(���ǩ�A�* 

Average reward per step��}�BV�       ��2	���ǩ�A�*

epsilon��}����.       ��W�	o���ǩ�A�* 

Average reward per step��}��Q!�       ��2	U���ǩ�A�*

epsilon��}�)8�.       ��W�	���ǩ�A�* 

Average reward per step��}�_�       ��2	����ǩ�A�*

epsilon��}��f��0       ���_	���ǩ�A*#
!
Average reward per episodeO�D��ێI.       ��W�	s���ǩ�A*!

total reward per episode   �(��.       ��W�	�@��ǩ�A�* 

Average reward per stepO�D�z5��       ��2	hA��ǩ�A�*

epsilonO�D����.       ��W�	���ǩ�A�* 

Average reward per stepO�D���F       ��2	՗��ǩ�A�*

epsilonO�D����.       ��W�	!���ǩ�A�* 

Average reward per stepO�D��I�       ��2	B���ǩ�A�*

epsilonO�D�1��a.       ��W�	���ǩ�A�* 

Average reward per stepO�D�}�I�       ��2	����ǩ�A�*

epsilonO�D�
ݙ�.       ��W�	D���ǩ�A�* 

Average reward per stepO�D�<?�       ��2	"���ǩ�A�*

epsilonO�D� ,��.       ��W�	����ǩ�A�* 

Average reward per stepO�D�)�|�       ��2	;���ǩ�A�*

epsilonO�D��6;3.       ��W�	J���ǩ�A�* 

Average reward per stepO�D��� �       ��2	)���ǩ�A�*

epsilonO�D��X1�.       ��W�	����ǩ�A�* 

Average reward per stepO�D��LtR       ��2	s���ǩ�A�*

epsilonO�D���
�.       ��W�	����ǩ�A�* 

Average reward per stepO�D��)-       ��2	����ǩ�A�*

epsilonO�D��a$`.       ��W�	���ǩ�A�* 

Average reward per stepO�D�H<�       ��2	����ǩ�A�*

epsilonO�D�ehM.       ��W�	����ǩ�A�* 

Average reward per stepO�D�
�JT       ��2	����ǩ�A�*

epsilonO�D�����.       ��W�	���ǩ�A�* 

Average reward per stepO�D�/�       ��2	���ǩ�A�*

epsilonO�D��
�.       ��W�	���ǩ�A�* 

Average reward per stepO�D��2^�       ��2	���ǩ�A�*

epsilonO�D��{.       ��W�	�)��ǩ�A�* 

Average reward per stepO�D��C"       ��2	|*��ǩ�A�*

epsilonO�D��y�'.       ��W�	�'��ǩ�A�* 

Average reward per stepO�D�59�       ��2	A(��ǩ�A�*

epsilonO�D�Z�X.       ��W�		6��ǩ�A�* 

Average reward per stepO�D�����       ��2	�6��ǩ�A�*

epsilonO�D���Q.       ��W�	MK��ǩ�A�* 

Average reward per stepO�D���`�       ��2	
L��ǩ�A�*

epsilonO�D��U��.       ��W�	P��ǩ�A�* 

Average reward per stepO�D��|       ��2	�P��ǩ�A�*

epsilonO�D��?��.       ��W�	���ǩ�A�* 

Average reward per stepO�D���       ��2	��ǩ�A�*

epsilonO�D�駲[.       ��W�	����ǩ�A�* 

Average reward per stepO�D���       ��2	c���ǩ�A�*

epsilonO�D��߅.       ��W�	����ǩ�A�* 

Average reward per stepO�D�Gs��       ��2	e���ǩ�A�*

epsilonO�D�j��@.       ��W�	����ǩ�A�* 

Average reward per stepO�D�=�W�       ��2	T���ǩ�A�*

epsilonO�D����.       ��W�	����ǩ�A�* 

Average reward per stepO�D��x%       ��2	����ǩ�A�*

epsilonO�D��'N.       ��W�	>��ǩ�A�* 

Average reward per stepO�D�A�G�       ��2	1��ǩ�A�*

epsilonO�D�����.       ��W�	c��ǩ�A�* 

Average reward per stepO�D��b^�       ��2	�c��ǩ�A�*

epsilonO�D�+���.       ��W�	�l��ǩ�A�* 

Average reward per stepO�D�����       ��2	�m��ǩ�A�*

epsilonO�D��3��.       ��W�	Dm��ǩ�A�* 

Average reward per stepO�D���m�       ��2	�m��ǩ�A�*

epsilonO�D��j��.       ��W�	� �ǩ�A�* 

Average reward per stepO�D����       ��2	�� �ǩ�A�*

epsilonO�D�h�e.       ��W�	��ǩ�A�* 

Average reward per stepO�D���]�       ��2	���ǩ�A�*

epsilonO�D���Τ.       ��W�	Z��ǩ�A�* 

Average reward per stepO�D��F�       ��2	=��ǩ�A�*

epsilonO�D��_��.       ��W�	���ǩ�A�* 

Average reward per stepO�D����+       ��2	c��ǩ�A�*

epsilonO�D����5.       ��W�	��ǩ�A�* 

Average reward per stepO�D���E       ��2	��ǩ�A�*

epsilonO�D��ɍ�.       ��W�	Ӣ	�ǩ�A�* 

Average reward per stepO�D����"       ��2	��	�ǩ�A�*

epsilonO�D� � .       ��W�	�
�ǩ�A�* 

Average reward per stepO�D�lp�T       ��2	(�ǩ�A�*

epsilonO�D�{t��.       ��W�	��ǩ�A�* 

Average reward per stepO�D�5��        ��2	��ǩ�A�*

epsilonO�D�I.       ��W�	U�ǩ�A�* 

Average reward per stepO�D�w	�5       ��2	/�ǩ�A�*

epsilonO�D��d�.       ��W�	�ǩ�A�* 

Average reward per stepO�D���d-       ��2	��ǩ�A�*

epsilonO�D����.       ��W�	_)�ǩ�A�* 

Average reward per stepO�D��o�F       ��2	�)�ǩ�A�*

epsilonO�D�����.       ��W�	�4�ǩ�A�* 

Average reward per stepO�D��M�c       ��2	�5�ǩ�A�*

epsilonO�D�Aq!I.       ��W�	�B�ǩ�A�* 

Average reward per stepO�D�t�-       ��2	�C�ǩ�A�*

epsilonO�D��w��.       ��W�	X�ǩ�A�* 

Average reward per stepO�D��e�       ��2	�X�ǩ�A�*

epsilonO�D�[��.       ��W�	ca�ǩ�A�* 

Average reward per stepO�D�Qg��       ��2	|b�ǩ�A�*

epsilonO�D�C�b�.       ��W�	�j�ǩ�A�* 

Average reward per stepO�D�&�\{       ��2	�k�ǩ�A�*

epsilonO�D�_l�).       ��W�	v��ǩ�A�* 

Average reward per stepO�D���T�       ��2	e��ǩ�A�*

epsilonO�D��3�'.       ��W�	ۋ!�ǩ�A�* 

Average reward per stepO�D�˦g�       ��2	Ō!�ǩ�A�*

epsilonO�D��R .       ��W�	�#�ǩ�A�* 

Average reward per stepO�D�;y`       ��2	��#�ǩ�A�*

epsilonO�D�J�
.       ��W�	�[%�ǩ�A�* 

Average reward per stepO�D��_t       ��2	�\%�ǩ�A�*

epsilonO�D��Pی.       ��W�	�['�ǩ�A�* 

Average reward per stepO�D�_���       ��2	�\'�ǩ�A�*

epsilonO�D���\.       ��W�	<i)�ǩ�A�* 

Average reward per stepO�D�*��&       ��2	�i)�ǩ�A�*

epsilonO�D�=�.       ��W�	Tn+�ǩ�A�* 

Average reward per stepO�D���+�       ��2	7o+�ǩ�A�*

epsilonO�D�^�R�.       ��W�	��-�ǩ�A�* 

Average reward per stepO�D����       ��2	I�-�ǩ�A�*

epsilonO�D�@P�0       ���_	ß-�ǩ�A*#
!
Average reward per episode��ÿ&W2�.       ��W�	w�-�ǩ�A*!

total reward per episode  ���).       ��W�	Y�1�ǩ�A�* 

Average reward per step��ÿ�@f�       ��2	�1�ǩ�A�*

epsilon��ÿ�~�.       ��W�	�3�ǩ�A�* 

Average reward per step��ÿ�M�       ��2	�3�ǩ�A�*

epsilon��ÿ)��.       ��W�	�5�ǩ�A�* 

Average reward per step��ÿf�       ��2	��5�ǩ�A�*

epsilon��ÿ:|��.       ��W�	��7�ǩ�A�* 

Average reward per step��ÿ�9$       ��2	��7�ǩ�A�*

epsilon��ÿpL��.       ��W�	��9�ǩ�A�* 

Average reward per step��ÿ�HUh       ��2	`�9�ǩ�A�*

epsilon��ÿ�w_.       ��W�	��;�ǩ�A�* 

Average reward per step��ÿC�A<       ��2	p�;�ǩ�A�*

epsilon��ÿI��r.       ��W�	}�=�ǩ�A�* 

Average reward per step��ÿ|}2p       ��2	J�=�ǩ�A�*

epsilon��ÿ���.       ��W�	l�?�ǩ�A�* 

Average reward per step��ÿq��5       ��2	>�?�ǩ�A�*

epsilon��ÿź�.       ��W�	7�A�ǩ�A�* 

Average reward per step��ÿc��       ��2	��A�ǩ�A�*

epsilon��ÿ��@�.       ��W�	�:C�ǩ�A�* 

Average reward per step��ÿ����       ��2	�;C�ǩ�A�*

epsilon��ÿq���.       ��W�	DNE�ǩ�A�* 

Average reward per step��ÿ�7�       ��2	�NE�ǩ�A�*

epsilon��ÿ�� 
.       ��W�	�QG�ǩ�A�* 

Average reward per step��ÿG}       ��2	PRG�ǩ�A�*

epsilon��ÿ�ok.       ��W�	9FI�ǩ�A�* 

Average reward per step��ÿ-�4       ��2	GI�ǩ�A�*

epsilon��ÿ'��?.       ��W�	�HK�ǩ�A�* 

Average reward per step��ÿ�N�q       ��2	�IK�ǩ�A�*

epsilon��ÿ�CV{.       ��W�	*SM�ǩ�A�* 

Average reward per step��ÿ�MJ       ��2	TM�ǩ�A�*

epsilon��ÿ~'�.       ��W�	-]O�ǩ�A�* 

Average reward per step��ÿ{�h       ��2	^O�ǩ�A�*

epsilon��ÿ���.       ��W�	�wQ�ǩ�A�* 

Average reward per step��ÿ�F
�       ��2	�xQ�ǩ�A�*

epsilon��ÿ�֬?.       ��W�	��S�ǩ�A�* 

Average reward per step��ÿWVH�       ��2	s�S�ǩ�A�*

epsilon��ÿ��.:.       ��W�	�U�ǩ�A�* 

Average reward per step��ÿMl�i       ��2	��U�ǩ�A�*

epsilon��ÿ�a}�.       ��W�	�W�ǩ�A�* 

Average reward per step��ÿ 1�q       ��2	��W�ǩ�A�*

epsilon��ÿ]��.       ��W�	|�Y�ǩ�A�* 

Average reward per step��ÿ�' A       ��2	M�Y�ǩ�A�*

epsilon��ÿk���.       ��W�	��[�ǩ�A�* 

Average reward per step��ÿ�M/�       ��2	(�[�ǩ�A�*

epsilon��ÿ~��`.       ��W�	�]�ǩ�A�* 

Average reward per step��ÿ�       ��2	��]�ǩ�A�*

epsilon��ÿ����.       ��W�	t�_�ǩ�A�* 

Average reward per step��ÿK��       ��2	,�_�ǩ�A�*

epsilon��ÿ	Z��.       ��W�	�Za�ǩ�A�* 

Average reward per step��ÿŉ"j       ��2	�[a�ǩ�A�*

epsilon��ÿBwr.       ��W�	�b�ǩ�A�* 

Average reward per step��ÿ<��7       ��2	��b�ǩ�A�*

epsilon��ÿ&��.       ��W�	K�d�ǩ�A�* 

Average reward per step��ÿ.��       ��2	��d�ǩ�A�*

epsilon��ÿ��.       ��W�	4�f�ǩ�A�* 

Average reward per step��ÿ%�h       ��2	�f�ǩ�A�*

epsilon��ÿr#�}.       ��W�	<�h�ǩ�A�* 

Average reward per step��ÿ%ͬ�       ��2	'�h�ǩ�A�*

epsilon��ÿ�թW.       ��W�	��j�ǩ�A�* 

Average reward per step��ÿ���       ��2	}�j�ǩ�A�*

epsilon��ÿT��.       ��W�	��l�ǩ�A�* 

Average reward per step��ÿsl�*       ��2	\�l�ǩ�A�*

epsilon��ÿ؜�<.       ��W�	��n�ǩ�A�* 

Average reward per step��ÿ~d>�       ��2	��n�ǩ�A�*

epsilon��ÿ�M�0       ���_	�o�ǩ�A*#
!
Average reward per episode  ��Q��.       ��W�	l	o�ǩ�A*!

total reward per episode  	���6�.       ��W�	�@s�ǩ�A�* 

Average reward per step  ������       ��2	%As�ǩ�A�*

epsilon  ��OE��.       ��W�	�[u�ǩ�A�* 

Average reward per step  ���=��       ��2	S\u�ǩ�A�*

epsilon  ���Rr\.       ��W�	caw�ǩ�A�* 

Average reward per step  ����u�       ��2	 bw�ǩ�A�*

epsilon  ��鋅.       ��W�	�ly�ǩ�A�* 

Average reward per step  ��;k:I       ��2	�my�ǩ�A�*

epsilon  ��aq��.       ��W�	�`{�ǩ�A�* 

Average reward per step  ���&r       ��2	�a{�ǩ�A�*

epsilon  ��܂�.       ��W�	�b}�ǩ�A�* 

Average reward per step  ����1       ��2	�c}�ǩ�A�*

epsilon  ��D@+.       ��W�	�_�ǩ�A�* 

Average reward per step  �� ��       ��2	t`�ǩ�A�*

epsilon  ����i.       ��W�	g��ǩ�A�* 

Average reward per step  �����q       ��2	�g��ǩ�A�*

epsilon  ����B.       ��W�	�\��ǩ�A�* 

Average reward per step  ��$�n�       ��2	y]��ǩ�A�*

epsilon  ��E���.       ��W�	#f��ǩ�A�* 

Average reward per step  ��Z�[       ��2	�f��ǩ�A�*

epsilon  ����;L.       ��W�	CU��ǩ�A�* 

Average reward per step  ��)�.n       ��2	�U��ǩ�A�*

epsilon  ����L.       ��W�	�S��ǩ�A�* 

Average reward per step  ��W���       ��2	?T��ǩ�A�*

epsilon  ��Us;
.       ��W�	eU��ǩ�A�* 

Average reward per step  ������       ��2	V��ǩ�A�*

epsilon  ����.       ��W�	����ǩ�A�* 

Average reward per step  ���7�       ��2	Ϊ��ǩ�A�*

epsilon  ��'C�G.       ��W�	����ǩ�A�* 

Average reward per step  �����       ��2	q���ǩ�A�*

epsilon  �����.       ��W�	l���ǩ�A�* 

Average reward per step  ��"��?       ��2	����ǩ�A�*

epsilon  ��Dxob.       ��W�	=���ǩ�A�* 

Average reward per step  ������       ��2	Զ��ǩ�A�*

epsilon  ����.       ��W�	�Ӕ�ǩ�A�* 

Average reward per step  ���H�       ��2	�Ԕ�ǩ�A�*

epsilon  ��ʒ,�.       ��W�	x��ǩ�A�* 

Average reward per step  ����[�       ��2	_��ǩ�A�*

epsilon  ������.       ��W�	����ǩ�A�* 

Average reward per step  ��$b       ��2	����ǩ�A�*

epsilon  �����&.       ��W�	���ǩ�A�* 

Average reward per step  ��m|H�       ��2	1	��ǩ�A�*

epsilon  ��\O�.       ��W�	���ǩ�A�* 

Average reward per step  ��Ԁ'       ��2	n��ǩ�A�*

epsilon  ���#.       ��W�	|&��ǩ�A�* 

Average reward per step  ��}r�       ��2	'��ǩ�A�*

epsilon  ��n/�.       ��W�	>��ǩ�A�* 

Average reward per step  ���	�S       ��2	�>��ǩ�A�*

epsilon  ��!�:�.       ��W�	7���ǩ�A�* 

Average reward per step  ��-p9�       ��2	Ω��ǩ�A�*

epsilon  ���e8�.       ��W�	���ǩ�A�* 

Average reward per step  �����       ��2	��ǩ�A�*

epsilon  ���	�s.       ��W�	���ǩ�A�* 

Average reward per step  ���4;e       ��2	K��ǩ�A�*

epsilon  ����T.       ��W�	���ǩ�A�* 

Average reward per step  ��GTqm       ��2	T��ǩ�A�*

epsilon  ��[��.       ��W�	2��ǩ�A�* 

Average reward per step  ���x�_       ��2	��ǩ�A�*

epsilon  ���uȻ.       ��W�	�2��ǩ�A�* 

Average reward per step  ��\���       ��2	�3��ǩ�A�*

epsilon  ����90       ���_	/P��ǩ�A*#
!
Average reward per episode  ��LQ�.       ��W�	�P��ǩ�A*!

total reward per episode  ����.       ��W�	�O��ǩ�A�* 

Average reward per step  ���^`�       ��2	�P��ǩ�A�*

epsilon  ��2~�.       ��W�		m��ǩ�A�* 

Average reward per step  ��t"�n       ��2	�m��ǩ�A�*

epsilon  ���g�S.       ��W�	�n��ǩ�A�* 

Average reward per step  ����N       ��2	ro��ǩ�A�*

epsilon  �����.       ��W�	��ǩ�A�* 

Average reward per step  ����Х       ��2	���ǩ�A�*

epsilon  ���䝹.       ��W�	�!��ǩ�A�* 

Average reward per step  ��9�       ��2	�"��ǩ�A�*

epsilon  ���fl.       ��W�	?���ǩ�A�* 

Average reward per step  �����p       ��2	���ǩ�A�*

epsilon  ������.       ��W�	υ��ǩ�A�* 

Average reward per step  ���L��       ��2	b���ǩ�A�*

epsilon  ��`�E�.       ��W�	����ǩ�A�* 

Average reward per step  ���t       ��2	J���ǩ�A�*

epsilon  ��:݊�.       ��W�	$���ǩ�A�* 

Average reward per step  ��|Z֐       ��2	���ǩ�A�*

epsilon  ���L.       ��W�	����ǩ�A�* 

Average reward per step  ��A\�v       ��2	����ǩ�A�*

epsilon  ��V�� .       ��W�	����ǩ�A�* 

Average reward per step  ��r2M       ��2	����ǩ�A�*

epsilon  ���HH.       ��W�	_���ǩ�A�* 

Average reward per step  ����.�       ��2	_���ǩ�A�*

epsilon  ���hJ.       ��W�	f���ǩ�A�* 

Average reward per step  ����]D       ��2	���ǩ�A�*

epsilon  ��wH_.       ��W�	���ǩ�A�* 

Average reward per step  ���+�       ��2	i��ǩ�A�*

epsilon  ��O�o.       ��W�	���ǩ�A�* 

Average reward per step  ����       ��2	���ǩ�A�*

epsilon  ��:�L�.       ��W�	/��ǩ�A�* 

Average reward per step  ���aT       ��2	���ǩ�A�*

epsilon  ��Y�6�.       ��W�	�A��ǩ�A�* 

Average reward per step  ���v�       ��2	�B��ǩ�A�*

epsilon  ��q@�s.       ��W�	�`��ǩ�A�* 

Average reward per step  ��N��N       ��2	�a��ǩ�A�*

epsilon  ��KK�P.       ��W�	�b��ǩ�A�* 

Average reward per step  ��+�       ��2	xc��ǩ�A�*

epsilon  ���Nj.       ��W�	�|��ǩ�A�* 

Average reward per step  ����_       ��2	�}��ǩ�A�*

epsilon  ������.       ��W�	Wy��ǩ�A�* 

Average reward per step  ��K8Q       ��2	%z��ǩ�A�*

epsilon  ����!.       ��W�	r��ǩ�A�* 

Average reward per step  ��[��q       ��2	�r��ǩ�A�*

epsilon  ���O�n.       ��W�	�{��ǩ�A�* 

Average reward per step  ��K�/       ��2	�|��ǩ�A�*

epsilon  ����{�.       ��W�	׆��ǩ�A�* 

Average reward per step  ���Ʊ       ��2	����ǩ�A�*

epsilon  ��G0P.       ��W�	����ǩ�A�* 

Average reward per step  ������       ��2	����ǩ�A�*

epsilon  ���
�.       ��W�	g��ǩ�A�* 

Average reward per step  ��/j߇       ��2	��ǩ�A�*

epsilon  ��.F�9.       ��W�	+m��ǩ�A�* 

Average reward per step  ��x�       ��2	�m��ǩ�A�*

epsilon  ���z=.       ��W�	�t��ǩ�A�* 

Average reward per step  ��t�W/       ��2	qu��ǩ�A�*

epsilon  ��Q-�.       ��W�	�x��ǩ�A�* 

Average reward per step  ���_�       ��2	ly��ǩ�A�*

epsilon  ��S��&.       ��W�	{��ǩ�A�* 

Average reward per step  �����,       ��2	�{��ǩ�A�*

epsilon  ����.       ��W�	0���ǩ�A�* 

Average reward per step  ��Z�<�       ��2	˄��ǩ�A�*

epsilon  �����.       ��W�	"���ǩ�A�* 

Average reward per step  ��ۜ�       ��2	Ō��ǩ�A�*

epsilon  �����&.       ��W�	E��ǩ�A�* 

Average reward per step  ����K       ��2	���ǩ�A�*

epsilon  ����eF.       ��W�	*���ǩ�A�* 

Average reward per step  ��3o=�       ��2	���ǩ�A�*

epsilon  ��sfh�.       ��W�	J���ǩ�A�* 

Average reward per step  ����@       ��2	ݘ��ǩ�A�*

epsilon  ��dv�,.       ��W�	���ǩ�A�* 

Average reward per step  ��S޷       ��2	����ǩ�A�*

epsilon  ��[t�g.       ��W�	u���ǩ�A�* 

Average reward per step  ��o�n       ��2	T���ǩ�A�*

epsilon  ����5.       ��W�	6���ǩ�A�* 

Average reward per step  ���E��       ��2	Ҭ��ǩ�A�*

epsilon  ��qkt�.       ��W�	6���ǩ�A�* 

Average reward per step  �� hU       ��2	���ǩ�A�*

epsilon  ���׮.       ��W�	����ǩ�A�* 

Average reward per step  ��,��F       ��2	����ǩ�A�*

epsilon  ���醢.       ��W�	<� �ǩ�A�* 

Average reward per step  ��2@	       ��2	� �ǩ�A�*

epsilon  �� ���.       ��W�	��ǩ�A�* 

Average reward per step  ��m�^�       ��2	���ǩ�A�*

epsilon  ���?��.       ��W�	��ǩ�A�* 

Average reward per step  ���,>�       ��2	��ǩ�A�*

epsilon  ��6��Z.       ��W�	���ǩ�A�* 

Average reward per step  ���ɱ�       ��2	h��ǩ�A�*

epsilon  ���z�.       ��W�	��ǩ�A�* 

Average reward per step  ��oX2�       ��2	���ǩ�A�*

epsilon  ���W��.       ��W�		�
�ǩ�A�* 

Average reward per step  ��L��E       ��2	��
�ǩ�A�*

epsilon  ��=dÂ.       ��W�	���ǩ�A�* 

Average reward per step  ��syiu       ��2	0��ǩ�A�*

epsilon  ��	c�L.       ��W�	�(�ǩ�A�* 

Average reward per step  ����kb       ��2	t)�ǩ�A�*

epsilon  ��p۾�.       ��W�	5&�ǩ�A�* 

Average reward per step  �����"       ��2	$'�ǩ�A�*

epsilon  ����N.       ��W�	�<�ǩ�A�* 

Average reward per step  ���       ��2	�=�ǩ�A�*

epsilon  ��j��.       ��W�	�R�ǩ�A�* 

Average reward per step  ���8�!       ��2	�S�ǩ�A�*

epsilon  ���$�.       ��W�	vP�ǩ�A�* 

Average reward per step  ���8?�       ��2	Q�ǩ�A�*

epsilon  ��9E�V.       ��W�	RH�ǩ�A�* 

Average reward per step  ��kX�c       ��2	(I�ǩ�A�*

epsilon  ��H�O.       ��W�	7S�ǩ�A�* 

Average reward per step  �����       ��2	T�ǩ�A�*

epsilon  ��LD�.       ��W�	�T�ǩ�A�* 

Average reward per step  ��QD��       ��2	7U�ǩ�A�*

epsilon  ������.       ��W�	iW�ǩ�A�* 

Average reward per step  ��N���       ��2	X�ǩ�A�*

epsilon  ���5�.       ��W�	�p �ǩ�A�* 

Average reward per step  ���Y��       ��2	Pq �ǩ�A�*

epsilon  ��vR�.       ��W�	�v"�ǩ�A�* 

Average reward per step  ��z��;       ��2	lw"�ǩ�A�*

epsilon  �����n.       ��W�	�$�ǩ�A�* 

Average reward per step  ��Z�R=       ��2	��$�ǩ�A�*

epsilon  ��&H� .       ��W�	��&�ǩ�A�* 

Average reward per step  ����W       ��2	y�&�ǩ�A�*

epsilon  ���	�8.       ��W�	��(�ǩ�A�* 

Average reward per step  ����A�       ��2	��(�ǩ�A�*

epsilon  ��s�[.       ��W�	(�*�ǩ�A�* 

Average reward per step  ���+k\       ��2	�*�ǩ�A�*

epsilon  ����+&.       ��W�	�-�ǩ�A�* 

Average reward per step  ������       ��2	�-�ǩ�A�*

epsilon  ����h�0       ���_	�<-�ǩ�A*#
!
Average reward per episode��{���.       ��W�	�=-�ǩ�A*!

total reward per episode  x��L��.       ��W�	A�0�ǩ�A�* 

Average reward per step��{�gC�       ��2	ԛ0�ǩ�A�*

epsilon��{��D�.       ��W�	+�2�ǩ�A�* 

Average reward per step��{�&��       ��2	�2�ǩ�A�*

epsilon��{�潟�.       ��W�	��4�ǩ�A�* 

Average reward per step��{��}g�       ��2	f�4�ǩ�A�*

epsilon��{��f�.       ��W�	&�6�ǩ�A�* 

Average reward per step��{�@kVX       ��2	��6�ǩ�A�*

epsilon��{���.       ��W�	��8�ǩ�A�* 

Average reward per step��{�-�8q       ��2	?�8�ǩ�A�*

epsilon��{�u�.       ��W�	u:�ǩ�A�* 

Average reward per step��{�h��t       ��2	!:�ǩ�A�*

epsilon��{��a.       ��W�	<�ǩ�A�* 

Average reward per step��{��Dn       ��2	�<�ǩ�A�*

epsilon��{��.�.       ��W�	�,>�ǩ�A�* 

Average reward per step��{���~       ��2	�->�ǩ�A�*

epsilon��{�nߨ�.       ��W�	�(@�ǩ�A�* 

Average reward per step��{��4�       ��2	g)@�ǩ�A�*

epsilon��{���q.       ��W�	81B�ǩ�A�* 

Average reward per step��{�x�       ��2	�1B�ǩ�A�*

epsilon��{����.       ��W�	PVD�ǩ�A�* 

Average reward per step��{����       ��2	�VD�ǩ�A�*

epsilon��{�V��.       ��W�	sF�ǩ�A�* 

Average reward per step��{��N��       ��2	�sF�ǩ�A�*

epsilon��{��*\�.       ��W�	P�G�ǩ�A�* 

Average reward per step��{���       ��2	��G�ǩ�A�*

epsilon��{�t"�.       ��W�	^.I�ǩ�A�* 

Average reward per step��{��N~       ��2	+/I�ǩ�A�*

epsilon��{��m�.       ��W�	�sK�ǩ�A�* 

Average reward per step��{��:��       ��2	�tK�ǩ�A�*

epsilon��{���J.       ��W�	�yM�ǩ�A�* 

Average reward per step��{��p2.       ��2	}zM�ǩ�A�*

epsilon��{�$J�9.       ��W�	��N�ǩ�A�* 

Average reward per step��{�+�3�       ��2	<�N�ǩ�A�*

epsilon��{��B.       ��W�	�OP�ǩ�A�* 

Average reward per step��{�6O �       ��2	ePP�ǩ�A�*

epsilon��{���.       ��W�	�GR�ǩ�A�* 

Average reward per step��{�<i��       ��2	oHR�ǩ�A�*

epsilon��{�\�ȱ.       ��W�	]PT�ǩ�A�* 

Average reward per step��{�ꕰH       ��2	3QT�ǩ�A�*

epsilon��{�B�N�.       ��W�	wNV�ǩ�A�* 

Average reward per step��{�ѧ*I       ��2	QOV�ǩ�A�*

epsilon��{�L�L�.       ��W�	� Y�ǩ�A�* 

Average reward per step��{���       ��2	�Y�ǩ�A�*

epsilon��{���|�.       ��W�	jmZ�ǩ�A�* 

Average reward per step��{���       ��2	�mZ�ǩ�A�*

epsilon��{�t��.       ��W�	]o\�ǩ�A�* 

Average reward per step��{�tV��       ��2	3p\�ǩ�A�*

epsilon��{�QO>.       ��W�	6�]�ǩ�A�* 

Average reward per step��{����_       ��2	�]�ǩ�A�*

epsilon��{��P�.       ��W�	h�_�ǩ�A�* 

Average reward per step��{��?^�       ��2	�_�ǩ�A�*

epsilon��{�[t[�.       ��W�	��a�ǩ�A�* 

Average reward per step��{����        ��2	��a�ǩ�A�*

epsilon��{�#��v.       ��W�	��c�ǩ�A�* 

Average reward per step��{�)��@       ��2	T�c�ǩ�A�*

epsilon��{�V8~�.       ��W�	��e�ǩ�A�* 

Average reward per step��{��#�`       ��2	��e�ǩ�A�*

epsilon��{��"�z.       ��W�	��g�ǩ�A�* 

Average reward per step��{�yC��       ��2	��g�ǩ�A�*

epsilon��{���O.       ��W�	`�i�ǩ�A�* 

Average reward per step��{���       ��2	��i�ǩ�A�*

epsilon��{���/�.       ��W�	�k�ǩ�A�* 

Average reward per step��{�>oJ
       ��2	��k�ǩ�A�*

epsilon��{�7�$�.       ��W�	~�m�ǩ�A�* 

Average reward per step��{��Y�       ��2	�m�ǩ�A�*

epsilon��{���x2.       ��W�	o�ǩ�A�* 

Average reward per step��{����       ��2	�o�ǩ�A�*

epsilon��{���3�.       ��W�	�!q�ǩ�A�* 

Average reward per step��{��nH�       ��2	F"q�ǩ�A�*

epsilon��{�cX��.       ��W�	�s�ǩ�A�* 

Average reward per step��{�\���       ��2	�s�ǩ�A�*

epsilon��{����.       ��W�	�u�ǩ�A�* 

Average reward per step��{�Z���       ��2	�u�ǩ�A�*

epsilon��{�8�*.       ��W�	w�ǩ�A�* 

Average reward per step��{��W�       ��2	�w�ǩ�A�*

epsilon��{�*�O�.       ��W�	�y�ǩ�A�* 

Average reward per step��{�5�
�       ��2	�y�ǩ�A�*

epsilon��{����.       ��W�	� {�ǩ�A�* 

Average reward per step��{��R�       ��2	�!{�ǩ�A�*

epsilon��{�o~n�.       ��W�	�,}�ǩ�A�* 

Average reward per step��{��̸�       ��2	�-}�ǩ�A�*

epsilon��{�e���.       ��W�	��ǩ�A�* 

Average reward per step��{��o��       ��2	G��ǩ�A�*

epsilon��{����.       ��W�	ϻ��ǩ�A�* 

Average reward per step��{�e5N�       ��2	j���ǩ�A�*

epsilon��{��;�.       ��W�	����ǩ�A�* 

Average reward per step��{�7�        ��2	b���ǩ�A�*

epsilon��{����.       ��W�	5���ǩ�A�* 

Average reward per step��{��#=z       ��2	Ե��ǩ�A�*

epsilon��{�M �.       ��W�	����ǩ�A�* 

Average reward per step��{�
)O�       ��2	M���ǩ�A�*

epsilon��{�$'�].       ��W�	���ǩ�A�* 

Average reward per step��{�g�I       ��2	鷉�ǩ�A�*

epsilon��{�3:a�.       ��W�	�h��ǩ�A�* 

Average reward per step��{����I       ��2	�i��ǩ�A�*

epsilon��{�N���.       ��W�	5��ǩ�A�* 

Average reward per step��{�kD��       ��2	���ǩ�A�*

epsilon��{����.       ��W�	x���ǩ�A�* 

Average reward per step��{�C���       ��2	��ǩ�A�*

epsilon��{�2K��.       ��W�	_��ǩ�A�* 

Average reward per step��{��"4       ��2	A��ǩ�A�*

epsilon��{�𶄊.       ��W�	* ��ǩ�A�* 

Average reward per step��{��F��       ��2	 ��ǩ�A�*

epsilon��{��.       ��W�	L���ǩ�A�* 

Average reward per step��{�����       ��2	����ǩ�A�*

epsilon��{���h�.       ��W�	�c��ǩ�A�* 

Average reward per step��{��
C"       ��2	�d��ǩ�A�*

epsilon��{�/��.       ��W�	�i��ǩ�A�* 

Average reward per step��{�.���       ��2	8j��ǩ�A�*

epsilon��{��|�.       ��W�	�a��ǩ�A�* 

Average reward per step��{�C���       ��2	�b��ǩ�A�*

epsilon��{��O��.       ��W�	Ja��ǩ�A�* 

Average reward per step��{�^�n�       ��2	b��ǩ�A�*

epsilon��{���>�.       ��W�	�^��ǩ�A�* 

Average reward per step��{��h"W       ��2	�_��ǩ�A�*

epsilon��{���$�.       ��W�	�S��ǩ�A�* 

Average reward per step��{�/g�       ��2	�T��ǩ�A�*

epsilon��{����.       ��W�	LS��ǩ�A�* 

Average reward per step��{���SW       ��2		T��ǩ�A�*

epsilon��{���r.       ��W�	yW��ǩ�A�* 

Average reward per step��{�HωK       ��2	?X��ǩ�A�*

epsilon��{�s�e�.       ��W�	Q���ǩ�A�* 

Average reward per step��{����       ��2	����ǩ�A�*

epsilon��{���.       ��W�	���ǩ�A�* 

Average reward per step��{����       ��2	����ǩ�A�*

epsilon��{��'��.       ��W�	����ǩ�A�* 

Average reward per step��{��Cz�       ��2	����ǩ�A�*

epsilon��{��3}�.       ��W�	Y���ǩ�A�* 

Average reward per step��{�|�xf       ��2	*���ǩ�A�*

epsilon��{��婚.       ��W�	N���ǩ�A�* 

Average reward per step��{���T=       ��2	��ǩ�A�*

epsilon��{�m��.       ��W�	���ǩ�A�* 

Average reward per step��{�:P�       ��2	稰�ǩ�A�*

epsilon��{�|2a.       ��W�	D���ǩ�A�* 

Average reward per step��{���       ��2	���ǩ�A�*

epsilon��{���v.       ��W�	,���ǩ�A�* 

Average reward per step��{���       ��2	
���ǩ�A�*

epsilon��{���!0.       ��W�	k���ǩ�A�* 

Average reward per step��{��Xv*       ��2	R���ǩ�A�*

epsilon��{�y�c�.       ��W�	u��ǩ�A�* 

Average reward per step��{� �       ��2	��ǩ�A�*

epsilon��{���.       ��W�	���ǩ�A�* 

Average reward per step��{���k       ��2	���ǩ�A�*

epsilon��{����.       ��W�	%��ǩ�A�* 

Average reward per step��{��x       ��2	���ǩ�A�*

epsilon��{�_��.       ��W�	���ǩ�A�* 

Average reward per step��{��Po       ��2	}��ǩ�A�*

epsilon��{�]��H.       ��W�	P��ǩ�A�* 

Average reward per step��{����,       ��2	��ǩ�A�*

epsilon��{��f_#.       ��W�	D��ǩ�A�* 

Average reward per step��{����       ��2	��ǩ�A�*

epsilon��{�ʔ.       ��W�	����ǩ�A�* 

Average reward per step��{��'"w       ��2	W���ǩ�A�*

epsilon��{�J-L�.       ��W�	���ǩ�A�* 

Average reward per step��{�~#)       ��2	k��ǩ�A�*

epsilon��{���q�.       ��W�	~���ǩ�A�* 

Average reward per step��{�u��       ��2	L���ǩ�A�*

epsilon��{�M�.       ��W�	�Y��ǩ�A�* 

Average reward per step��{�ѡch       ��2	KZ��ǩ�A�*

epsilon��{��$�6.       ��W�	����ǩ�A�* 

Average reward per step��{�Aܒ�       ��2	E���ǩ�A�*

epsilon��{�Aۅ.       ��W�	����ǩ�A�* 

Average reward per step��{��^:C       ��2	����ǩ�A�*

epsilon��{�?F��.       ��W�	���ǩ�A�* 

Average reward per step��{��=91       ��2	��ǩ�A�*

epsilon��{��I[%.       ��W�	�|��ǩ�A�* 

Average reward per step��{�C\��       ��2	�}��ǩ�A�*

epsilon��{��z.       ��W�	@���ǩ�A�* 

Average reward per step��{����o       ��2	���ǩ�A�*

epsilon��{�2�..       ��W�	����ǩ�A�* 

Average reward per step��{���5�       ��2	7���ǩ�A�*

epsilon��{�Phޏ.       ��W�	���ǩ�A�* 

Average reward per step��{�=w        ��2	���ǩ�A�*

epsilon��{���.       ��W�	����ǩ�A�* 

Average reward per step��{�c��<       ��2	����ǩ�A�*

epsilon��{�]�@�.       ��W�	�>��ǩ�A�* 

Average reward per step��{��ɿ�       ��2	W?��ǩ�A�*

epsilon��{���Qb.       ��W�	ۋ��ǩ�A�* 

Average reward per step��{���g�       ��2	r���ǩ�A�*

epsilon��{�e�k.       ��W�	w���ǩ�A�* 

Average reward per step��{�ì�!       ��2	H���ǩ�A�*

epsilon��{����.       ��W�	���ǩ�A�* 

Average reward per step��{�73��       ��2	H���ǩ�A�*

epsilon��{����".       ��W�	V���ǩ�A�* 

Average reward per step��{�cұ~       ��2	���ǩ�A�*

epsilon��{�$Y��.       ��W�	q���ǩ�A�* 

Average reward per step��{�ܑ�       ��2	*���ǩ�A�*

epsilon��{����.       ��W�	T���ǩ�A�* 

Average reward per step��{�����       ��2	*���ǩ�A�*

epsilon��{��c3.       ��W�	���ǩ�A�* 

Average reward per step��{�̻�G       ��2	¦��ǩ�A�*

epsilon��{���#�.       ��W�	w���ǩ�A�* 

Average reward per step��{��@Oe       ��2	���ǩ�A�*

epsilon��{�G_A�.       ��W�	{���ǩ�A�* 

Average reward per step��{����       ��2	���ǩ�A�*

epsilon��{�rT'�.       ��W�	&���ǩ�A�* 

Average reward per step��{�ڃ��       ��2	 ��ǩ�A�*

epsilon��{�:o`�.       ��W�	���ǩ�A�* 

Average reward per step��{�ܪ~�       ��2	���ǩ�A�*

epsilon��{�*��.       ��W�	#��ǩ�A�* 

Average reward per step��{��
��       ��2	���ǩ�A�*

epsilon��{���Gx.       ��W�	Z��ǩ�A�* 

Average reward per step��{���       ��2	���ǩ�A�*

epsilon��{����^.       ��W�	���ǩ�A�* 

Average reward per step��{��V       ��2	G��ǩ�A�*

epsilon��{�i@�.       ��W�	� ��ǩ�A�* 

Average reward per step��{�M"E�       ��2	T��ǩ�A�*

epsilon��{�^&o�.       ��W�	H���ǩ�A�* 

Average reward per step��{���T�       ��2	����ǩ�A�*

epsilon��{�o�Ω.       ��W�	\��ǩ�A�* 

Average reward per step��{�&�|�       ��2	��ǩ�A�*

epsilon��{�b�(�.       ��W�	;��ǩ�A�* 

Average reward per step��{�
��       ��2	��ǩ�A�*

epsilon��{�@.�>.       ��W�	�5�ǩ�A�* 

Average reward per step��{��vPX       ��2	�6�ǩ�A�*

epsilon��{�j��.       ��W�	:�ǩ�A�* 

Average reward per step��{�C�R�       ��2	�:�ǩ�A�*

epsilon��{�I��.       ��W�	aP�ǩ�A�* 

Average reward per step��{�����       ��2	/Q�ǩ�A�*

epsilon��{��A�C.       ��W�	�}�ǩ�A�* 

Average reward per step��{��I��       ��2	�~�ǩ�A�*

epsilon��{���.       ��W�	w�	�ǩ�A�* 

Average reward per step��{���	�       ��2	^�	�ǩ�A�*

epsilon��{� �S.       ��W�	���ǩ�A�* 

Average reward per step��{��WT       ��2	���ǩ�A�*

epsilon��{�U�.�.       ��W�	��ǩ�A�* 

Average reward per step��{���`�       ��2	���ǩ�A�*

epsilon��{���P.       ��W�	!��ǩ�A�* 

Average reward per step��{�k��       ��2	���ǩ�A�*

epsilon��{��kg.       ��W�	�>�ǩ�A�* 

Average reward per step��{�|4�G       ��2	K?�ǩ�A�*

epsilon��{��:H�.       ��W�	���ǩ�A�* 

Average reward per step��{��s       ��2	���ǩ�A�*

epsilon��{��i�\.       ��W�	}��ǩ�A�* 

Average reward per step��{�����       ��2	F��ǩ�A�*

epsilon��{��Z�/.       ��W�	Ϡ�ǩ�A�* 

Average reward per step��{�BLhx       ��2	b��ǩ�A�*

epsilon��{����|.       ��W�	��ǩ�A�* 

Average reward per step��{��.��       ��2	
��ǩ�A�*

epsilon��{�G�l.       ��W�	���ǩ�A�* 

Average reward per step��{�Z��       ��2	T��ǩ�A�*

epsilon��{��aP&.       ��W�	F��ǩ�A�* 

Average reward per step��{���ˎ       ��2	��ǩ�A�*

epsilon��{�&�q-.       ��W�	��ǩ�A�* 

Average reward per step��{�9�*       ��2	���ǩ�A�*

epsilon��{����.       ��W�	�� �ǩ�A�* 

Average reward per step��{�SO�       ��2	!� �ǩ�A�*

epsilon��{��zp.       ��W�	��"�ǩ�A�* 

Average reward per step��{�E4�       ��2	��"�ǩ�A�*

epsilon��{��̖�0       ���_	m�"�ǩ�A*#
!
Average reward per episode
�#?���.       ��W�	��"�ǩ�A*!

total reward per episode  �BG@�b.       ��W�	��&�ǩ�A�* 

Average reward per step
�#?9Q       ��2	��&�ǩ�A�*

epsilon
�#?-�.       ��W�	��(�ǩ�A�* 

Average reward per step
�#?��*�       ��2	��(�ǩ�A�*

epsilon
�#?�_�>.       ��W�	��*�ǩ�A�* 

Average reward per step
�#?���]       ��2	`�*�ǩ�A�*

epsilon
�#?B.       ��W�	�,,�ǩ�A�* 

Average reward per step
�#?���       ��2	�-,�ǩ�A�*

epsilon
�#?�T,�.       ��W�	�'.�ǩ�A�* 

Average reward per step
�#?^��*       ��2	t(.�ǩ�A�*

epsilon
�#?�H��.       ��W�	W?0�ǩ�A�* 

Average reward per step
�#?��J       ��2	:@0�ǩ�A�*

epsilon
�#?&���.       ��W�	�42�ǩ�A�* 

Average reward per step
�#?��0       ��2	D52�ǩ�A�*

epsilon
�#?�[�.       ��W�	1?4�ǩ�A�* 

Average reward per step
�#?��/       ��2	�?4�ǩ�A�*

epsilon
�#?V��V.       ��W�	M6�ǩ�A�* 

Average reward per step
�#?�>L�       ��2	HN6�ǩ�A�*

epsilon
�#?�c5E.       ��W�	Vf8�ǩ�A�* 

Average reward per step
�#?� #�       ��2	,g8�ǩ�A�*

epsilon
�#?-Ӱ.       ��W�	�l:�ǩ�A�* 

Average reward per step
�#?S�s{       ��2	�m:�ǩ�A�*

epsilon
�#?4i�Q.       ��W�	��<�ǩ�A�* 

Average reward per step
�#?�dNr       ��2	��<�ǩ�A�*

epsilon
�#?i�G.       ��W�	�>�ǩ�A�* 

Average reward per step
�#?f�e       ��2	ׄ>�ǩ�A�*

epsilon
�#?<�.       ��W�	��@�ǩ�A�* 

Average reward per step
�#?���       ��2	�@�ǩ�A�*

epsilon
�#?��pH0       ���_	\�@�ǩ�A*#
!
Average reward per episoden�>����.       ��W�	�@�ǩ�A*!

total reward per episode  '�Oқ4.       ��W�	mC�ǩ�A�* 

Average reward per stepn�>�~�h�       ��2	�mC�ǩ�A�*

epsilonn�>�ב}�.       ��W�	�nE�ǩ�A�* 

Average reward per stepn�>��(�       ��2	&oE�ǩ�A�*

epsilonn�>�[	�Z.       ��W�	rG�ǩ�A�* 

Average reward per stepn�>�m�T       ��2	�rG�ǩ�A�*

epsilonn�>����.       ��W�	(}I�ǩ�A�* 

Average reward per stepn�>�ؑ�       ��2	�}I�ǩ�A�*

epsilonn�>�.3^q.       ��W�	��K�ǩ�A�* 

Average reward per stepn�>����       ��2	��K�ǩ�A�*

epsilonn�>��F��.       ��W�	�M�ǩ�A�* 

Average reward per stepn�>��       ��2	��M�ǩ�A�*

epsilonn�>��l�".       ��W�	J�O�ǩ�A�* 

Average reward per stepn�>�_2>�       ��2	�O�ǩ�A�*

epsilonn�>�9�e.       ��W�	Y�Q�ǩ�A�* 

Average reward per stepn�>���       ��2	L�Q�ǩ�A�*

epsilonn�>���.       ��W�	��S�ǩ�A�* 

Average reward per stepn�>��G˚       ��2	a�S�ǩ�A�*

epsilonn�>��ˑ.       ��W�	)�U�ǩ�A�* 

Average reward per stepn�>�/�|-       ��2	��U�ǩ�A�*

epsilonn�>�Y�P.       ��W�	��W�ǩ�A�* 

Average reward per stepn�>����Y       ��2	��W�ǩ�A�*

epsilonn�>�Οcj.       ��W�	�Z�ǩ�A�* 

Average reward per stepn�>���Qp       ��2	�Z�ǩ�A�*

epsilonn�>�h�k5.       ��W�	�\�ǩ�A�* 

Average reward per stepn�>�.�^       ��2	�\�ǩ�A�*

epsilonn�>��B�.       ��W�	��]�ǩ�A�* 

Average reward per stepn�>�A`��       ��2	4�]�ǩ�A�*

epsilonn�>��{�.       ��W�	C�_�ǩ�A�* 

Average reward per stepn�>��">�       ��2	" `�ǩ�A�*

epsilonn�>�x�|.       ��W�	�,b�ǩ�A�* 

Average reward per stepn�>�,�h�       ��2	�-b�ǩ�A�*

epsilonn�>��F�j.       ��W�	�+d�ǩ�A�* 

Average reward per stepn�>�T��       ��2	w,d�ǩ�A�*

epsilonn�>�� �.       ��W�	�*f�ǩ�A�* 

Average reward per stepn�>�W�       ��2	s+f�ǩ�A�*

epsilonn�>�},&=.       ��W�	�4h�ǩ�A�* 

Average reward per stepn�>�f^��       ��2	a5h�ǩ�A�*

epsilonn�>��t�.       ��W�	�Ej�ǩ�A�* 

Average reward per stepn�>���       ��2	�Fj�ǩ�A�*

epsilonn�>���.       ��W�	Pl�ǩ�A�* 

Average reward per stepn�>���0       ��2	3Ql�ǩ�A�*

epsilonn�>� m��.       ��W�	wfn�ǩ�A�* 

Average reward per stepn�>�f�~J       ��2	Qgn�ǩ�A�*

epsilonn�>��lZ.       ��W�	a�p�ǩ�A�* 

Average reward per stepn�>�+�6�       ��2	D�p�ǩ�A�*

epsilonn�>��r�30       ���_	��p�ǩ�A*#
!
Average reward per episode-d���Ҝ.       ��W�	7�p�ǩ�A*!

total reward per episode  "���P.       ��W�	I,t�ǩ�A�* 

Average reward per step-d��&�v       ��2	-t�ǩ�A�*

epsilon-d����W�.       ��W�	�4v�ǩ�A�* 

Average reward per step-d��M��       ��2	e5v�ǩ�A�*

epsilon-d��9�f.       ��W�	�6x�ǩ�A�* 

Average reward per step-d��+"%v       ��2	a7x�ǩ�A�*

epsilon-d��)n��.       ��W�	�qz�ǩ�A�* 

Average reward per step-d����>[       ��2	�rz�ǩ�A�*

epsilon-d���pL�.       ��W�	ɐ|�ǩ�A�* 

Average reward per step-d�����       ��2	h�|�ǩ�A�*

epsilon-d��-B�.       ��W�	��~�ǩ�A�* 

Average reward per step-d���R�       ��2	��~�ǩ�A�*

epsilon-d��׉\.       ��W�	�ǩ�A�* 

Average reward per step-d����T�       ��2	����ǩ�A�*

epsilon-d���q��.       ��W�	���ǩ�A�* 

Average reward per step-d��~7'�       ��2	����ǩ�A�*

epsilon-d��m�C.       ��W�	����ǩ�A�* 

Average reward per step-d��:�N       ��2	L���ǩ�A�*

epsilon-d�����.       ��W�	a���ǩ�A�* 

Average reward per step-d������       ��2	3���ǩ�A�*

epsilon-d��ُ.       ��W�	���ǩ�A�* 

Average reward per step-d��&Y��       ��2	����ǩ�A�*

epsilon-d��($�.       ��W�	���ǩ�A�* 

Average reward per step-d����u�       ��2	:��ǩ�A�*

epsilon-d�����.       ��W�	K���ǩ�A�* 

Average reward per step-d����b�       ��2	ސ��ǩ�A�*

epsilon-d���N�.       ��W�	����ǩ�A�* 

Average reward per step-d���\!!       ��2	R���ǩ�A�*

epsilon-d��K֩�.       ��W�	.���ǩ�A�* 

Average reward per step-d��t���       ��2	����ǩ�A�*

epsilon-d��Z��|.       ��W�	ɓ�ǩ�A�* 

Average reward per step-d���D�C       ��2	 ʓ�ǩ�A�*

epsilon-d���A>.       ��W�	*7��ǩ�A�* 

Average reward per step-d��:�n       ��2	�7��ǩ�A�*

epsilon-d��\6��.       ��W�	9c��ǩ�A�* 

Average reward per step-d����|a       ��2	d��ǩ�A�*

epsilon-d��̙�L.       ��W�	l^��ǩ�A�* 

Average reward per step-d��`�       ��2	_��ǩ�A�*

epsilon-d���o35.       ��W�	Uh��ǩ�A�* 

Average reward per step-d��4{�o       ��2	ni��ǩ�A�*

epsilon-d��9���.       ��W�	���ǩ�A�* 

Average reward per step-d���to       ��2	����ǩ�A�*

epsilon-d��i��.       ��W�	^���ǩ�A�* 

Average reward per step-d��au��       ��2	����ǩ�A�*

epsilon-d��o˄G.       ��W�	z���ǩ�A�* 

Average reward per step-d����H       ��2	����ǩ�A�*

epsilon-d�����.       ��W�	��ǩ�A�* 

Average reward per step-d��^�L       ��2	����ǩ�A�*

epsilon-d��ܜ�_.       ��W�	�إ�ǩ�A�* 

Average reward per step-d�����       ��2	b٥�ǩ�A�*

epsilon-d��;��.       ��W�	���ǩ�A�* 

Average reward per step-d����1       ��2	���ǩ�A�*

epsilon-d���o�.       ��W�	���ǩ�A�* 

Average reward per step-d��xi�u       ��2	���ǩ�A�*

epsilon-d�����.       ��W�	���ǩ�A�* 

Average reward per step-d���|{�       ��2	j��ǩ�A�*

epsilon-d����IL.       ��W�	���ǩ�A�* 

Average reward per step-d��}_��       ��2	7��ǩ�A�*

epsilon-d��X[�A.       ��W�	2��ǩ�A�* 

Average reward per step-d��ង       ��2	�2��ǩ�A�*

epsilon-d��;M
�.       ��W�	�;��ǩ�A�* 

Average reward per step-d��\�'.       ��2	�<��ǩ�A�*

epsilon-d��>2��.       ��W�	l���ǩ�A�* 

Average reward per step-d�����       ��2	���ǩ�A�*

epsilon-d����B�.       ��W�	��ǩ�A�* 

Average reward per step-d���H�       ��2	���ǩ�A�*

epsilon-d���ى#.       ��W�	X ��ǩ�A�* 

Average reward per step-d��5�       ��2	� ��ǩ�A�*

epsilon-d������.       ��W�	�+��ǩ�A�* 

Average reward per step-d����       ��2	8,��ǩ�A�*

epsilon-d���.Z�.       ��W�	y?��ǩ�A�* 

Average reward per step-d�����"       ��2	h@��ǩ�A�*

epsilon-d���A�.       ��W�	EF��ǩ�A�* 

Average reward per step-d������       ��2	�F��ǩ�A�*

epsilon-d���.       ��W�	�K��ǩ�A�* 

Average reward per step-d��b��       ��2	8L��ǩ�A�*

epsilon-d����.       ��W�	ZG��ǩ�A�* 

Average reward per step-d��H�4�       ��2	�G��ǩ�A�*

epsilon-d���Ə�.       ��W�	�Z��ǩ�A�* 

Average reward per step-d��&�       ��2	6[��ǩ�A�*

epsilon-d���|�.       ��W�	�%��ǩ�A�* 

Average reward per step-d��.�4J       ��2	x&��ǩ�A�*

epsilon-d���0.       ��W�	*���ǩ�A�* 

Average reward per step-d��a�C       ��2	����ǩ�A�*

epsilon-d���x�.       ��W�	����ǩ�A�* 

Average reward per step-d��m�}       ��2	Â��ǩ�A�*

epsilon-d��U
v.       ��W�	9���ǩ�A�* 

Average reward per step-d���P��       ��2	Ԙ��ǩ�A�*

epsilon-d��TV.       ��W�	ƥ��ǩ�A�* 

Average reward per step-d��L��^       ��2	e���ǩ�A�*

epsilon-d���Kz�0       ���_	���ǩ�A*#
!
Average reward per episode9��XV�.       ��W�	����ǩ�A*!

total reward per episode  �¨ߏ2.       ��W�	#��ǩ�A�* 

Average reward per step9��2��       ��2	�#��ǩ�A�*

epsilon9����o.       ��W�	v7��ǩ�A�* 

Average reward per step9��mp�       ��2	8��ǩ�A�*

epsilon9��	]PL.       ��W�	�O��ǩ�A�* 

Average reward per step9���ss5       ��2	vP��ǩ�A�*

epsilon9�����$.       ��W�	����ǩ�A�* 

Average reward per step9�㿫���       ��2	:���ǩ�A�*

epsilon9����-.       ��W�	x��ǩ�A�* 

Average reward per step9��:��       ��2	��ǩ�A�*

epsilon9�㿃uo.       ��W�	s��ǩ�A�* 

Average reward per step9��*��       ��2	D��ǩ�A�*

epsilon9�㿜
	\.       ��W�	�$��ǩ�A�* 

Average reward per step9�㿋��F       ��2	[%��ǩ�A�*

epsilon9��g�l.       ��W�	G��ǩ�A�* 

Average reward per step9��"obl       ��2	�G��ǩ�A�*

epsilon9������.       ��W�	$b��ǩ�A�* 

Average reward per step9��V��m       ��2	c��ǩ�A�*

epsilon9��}X!�.       ��W�	����ǩ�A�* 

Average reward per step9�����       ��2	G���ǩ�A�*

epsilon9��n<dp.       ��W�	o���ǩ�A�* 

Average reward per step9��WL�U       ��2	R���ǩ�A�*

epsilon9����9�.       ��W�	���ǩ�A�* 

Average reward per step9���^��       ��2	g��ǩ�A�*

epsilon9����{.       ��W�	�z��ǩ�A�* 

Average reward per step9��ph`�       ��2	g{��ǩ�A�*

epsilon9���c�S.       ��W�	����ǩ�A�* 

Average reward per step9��K�ܤ       ��2	v���ǩ�A�*

epsilon9��i�&.       ��W�	����ǩ�A�* 

Average reward per step9��G$��       ��2	����ǩ�A�*

epsilon9����I�.       ��W�	ߦ��ǩ�A�* 

Average reward per step9�㿫���       ��2	v���ǩ�A�*

epsilon9��vwV�.       ��W�	���ǩ�A�* 

Average reward per step9���       ��2	ȳ��ǩ�A�*

epsilon9��-r�e.       ��W�	����ǩ�A�* 

Average reward per step9�㿂v(       ��2	����ǩ�A�*

epsilon9�㿮2�p.       ��W�	����ǩ�A�* 

Average reward per step9���1�#       ��2	<���ǩ�A�*

epsilon9��G�!.       ��W�	����ǩ�A�* 

Average reward per step9�����       ��2	v���ǩ�A�*

epsilon9�㿊�#.       ��W�	I���ǩ�A�* 

Average reward per step9�㿍���       ��2	���ǩ�A�*

epsilon9��~�D�.       ��W�	����ǩ�A�* 

Average reward per step9�㿂�Ez       ��2	����ǩ�A�*

epsilon9�㿌�..       ��W�	����ǩ�A�* 

Average reward per step9�㿮8�       ��2	L���ǩ�A�*

epsilon9�� �c�.       ��W�	� �ǩ�A�* 

Average reward per step9�㿤g�t       ��2	� �ǩ�A�*

epsilon9��xY&N.       ��W�	r�ǩ�A�* 

Average reward per step9��}܈       ��2	7�ǩ�A�*

epsilon9��Y�..       ��W�	C �ǩ�A�* 

Average reward per step9��@&�       ��2	q!�ǩ�A�*

epsilon9�㿬��.       ��W�	W]�ǩ�A�* 

Average reward per step9���� K       ��2	^�ǩ�A�*

epsilon9��:��g.       ��W�	d��ǩ�A�* 

Average reward per step9�㿙�F       ��2	��ǩ�A�*

epsilon9����e�0       ���_	��ǩ�A*#
!
Average reward per episode۶��a���.       ��W�	2�ǩ�A*!

total reward per episode  ����:.       ��W�	�:�ǩ�A�* 

Average reward per step۶��%��       ��2	C;�ǩ�A�*

epsilon۶��m_!a.       ��W�	SY�ǩ�A�* 

Average reward per step۶���T�       ��2	�Y�ǩ�A�*

epsilon۶��c�ٖ.       ��W�	�\�ǩ�A�* 

Average reward per step۶����       ��2	l]�ǩ�A�*

epsilon۶���% .       ��W�	�o�ǩ�A�* 

Average reward per step۶��8���       ��2	mp�ǩ�A�*

epsilon۶��\Z:.       ��W�	ʇ�ǩ�A�* 

Average reward per step۶������       ��2	]��ǩ�A�*

epsilon۶��ex��.       ��W�	z��ǩ�A�* 

Average reward per step۶�����X       ��2	��ǩ�A�*

epsilon۶���'g..       ��W�	���ǩ�A�* 

Average reward per step۶�����       ��2	:��ǩ�A�*

epsilon۶��7+�.       ��W�	*�ǩ�A�* 

Average reward per step۶��$O�-       ��2	��ǩ�A�*

epsilon۶��7�[.       ��W�	�-�ǩ�A�* 

Average reward per step۶������       ��2	f.�ǩ�A�*

epsilon۶��[�j0.       ��W�	���ǩ�A�* 

Average reward per step۶��z?�/       ��2	X��ǩ�A�*

epsilon۶���<۬.       ��W�	���ǩ�A�* 

Average reward per step۶��4�V       ��2	���ǩ�A�*

epsilon۶���e�.       ��W�	Z�!�ǩ�A�* 

Average reward per step۶��
�B       ��2	4�!�ǩ�A�*

epsilon۶���C�@.       ��W�	غ#�ǩ�A�* 

Average reward per step۶����k�       ��2	w�#�ǩ�A�*

epsilon۶��e/Y.       ��W�	��%�ǩ�A�* 

Average reward per step۶������       ��2	e�%�ǩ�A�*

epsilon۶��Q�W.       ��W�	H�'�ǩ�A�* 

Average reward per step۶�����/       ��2	��'�ǩ�A�*

epsilon۶��x��.       ��W�	�)�ǩ�A�* 

Average reward per step۶���܉<       ��2	��)�ǩ�A�*

epsilon۶����J�.       ��W�	b�+�ǩ�A�* 

Average reward per step۶���:�8       ��2	3�+�ǩ�A�*

epsilon۶���*V�.       ��W�	��-�ǩ�A�* 

Average reward per step۶��l�&w       ��2	��-�ǩ�A�*

epsilon۶����e.       ��W�	�/�ǩ�A�* 

Average reward per step۶�����i       ��2	��/�ǩ�A�*

epsilon۶���.       ��W�	�2�ǩ�A�* 

Average reward per step۶�����_       ��2	b2�ǩ�A�*

epsilon۶������.       ��W�	U4�ǩ�A�* 

Average reward per step۶��v\-m       ��2	�4�ǩ�A�*

epsilon۶���/�-.       ��W�	j16�ǩ�A�* 

Average reward per step۶���K�       ��2	n26�ǩ�A�*

epsilon۶����0       ���_	@N6�ǩ�A*#
!
Average reward per episode����%f.       ��W�	�N6�ǩ�A*!

total reward per episode  ���,.       ��W�	��9�ǩ�A�* 

Average reward per step������       ��2	/�9�ǩ�A�*

epsilon���t ǣ.       ��W�	��;�ǩ�A�* 

Average reward per step����x�       ��2	Z�;�ǩ�A�*

epsilon���|(l.       ��W�	(�=�ǩ�A�* 

Average reward per step���WWL�       ��2	��=�ǩ�A�*

epsilon�����Ga.       ��W�	T6@�ǩ�A�* 

Average reward per step����R�r       ��2	7@�ǩ�A�*

epsilon����k` .       ��W�	�=B�ǩ�A�* 

Average reward per step�������       ��2	}>B�ǩ�A�*

epsilon������.       ��W�	^JD�ǩ�A�* 

Average reward per step���6�B�       ��2	IKD�ǩ�A�*

epsilon���6�PT.       ��W�	fNF�ǩ�A�* 

Average reward per step�����m_       ��2	�NF�ǩ�A�*

epsilon���=X�.       ��W�	�PH�ǩ�A�* 

Average reward per step����V�       ��2	+QH�ǩ�A�*

epsilon���3��.       ��W�	�LJ�ǩ�A�* 

Average reward per step���$�       ��2	nMJ�ǩ�A�*

epsilon����l%�.       ��W�	}L�ǩ�A�* 

Average reward per step����S�       ��2	�}L�ǩ�A�*

epsilon�����.       ��W�	E�N�ǩ�A�* 

Average reward per step������       ��2	�N�ǩ�A�*

epsilon���y	�.       ��W�	��P�ǩ�A�* 

Average reward per step����Ṩ       ��2	��P�ǩ�A�*

epsilon���R]G�.       ��W�	��R�ǩ�A�* 

Average reward per step���N�kt       ��2	f�R�ǩ�A�*

epsilon�����+�.       ��W�	e�T�ǩ�A�* 

Average reward per step����	�       ��2	��T�ǩ�A�*

epsilon���g�#�.       ��W�	ϞV�ǩ�A�* 

Average reward per step����
I        ��2	��V�ǩ�A�*

epsilon����.       ��W�	�X�ǩ�A�* 

Average reward per step�����r6       ��2	��X�ǩ�A�*

epsilon����%.       ��W�	��Z�ǩ�A�* 

Average reward per step����F�       ��2	`�Z�ǩ�A�*

epsilon���Y��<.       ��W�	��\�ǩ�A�* 

Average reward per step���m���       ��2	ü\�ǩ�A�*

epsilon���*B��.       ��W�	��^�ǩ�A�* 

Average reward per step������!       ��2	�^�ǩ�A�*

epsilon���C�6.       ��W�	*�`�ǩ�A�* 

Average reward per step������       ��2	�`�ǩ�A�*

epsilon����a`�.       ��W�	4�b�ǩ�A�* 

Average reward per step����.��       ��2	��b�ǩ�A�*

epsilon���B�{�.       ��W�	��d�ǩ�A�* 

Average reward per step������>       ��2	u�d�ǩ�A�*

epsilon���M�3l.       ��W�	�f�ǩ�A�* 

Average reward per step���ۇ�       ��2	��f�ǩ�A�*

epsilon���m��.       ��W�	��h�ǩ�A�* 

Average reward per step���B��       ��2	d�h�ǩ�A�*

epsilon�������.       ��W�	��j�ǩ�A�* 

Average reward per step�����$       ��2	}�j�ǩ�A�*

epsilon���}��c.       ��W�	%m�ǩ�A�* 

Average reward per step���E�S�       ��2	�m�ǩ�A�*

epsilon�����G,.       ��W�	Q�n�ǩ�A�* 

Average reward per step������       ��2	'�n�ǩ�A�*

epsilon�����.       ��W�	Q�p�ǩ�A�* 

Average reward per step�����#�       ��2	��p�ǩ�A�*

epsilon����1Q.       ��W�	fs�ǩ�A�* 

Average reward per step���)'�       ��2	s�ǩ�A�*

epsilon���>Ƹ.       ��W�	�.u�ǩ�A�* 

Average reward per step���3���       ��2	4/u�ǩ�A�*

epsilon���4�{e.       ��W�	�'w�ǩ�A�* 

Average reward per step�����`       ��2	�(w�ǩ�A�*

epsilon���Y��.       ��W�	0+y�ǩ�A�* 

Average reward per step���x�ku       ��2	�+y�ǩ�A�*

epsilon�����n.       ��W�	�?{�ǩ�A�* 

Average reward per step�����M�       ��2	l@{�ǩ�A�*

epsilon���a�� .       ��W�	#L}�ǩ�A�* 

Average reward per step����`��       ��2	�L}�ǩ�A�*

epsilon���d��.       ��W�	�?�ǩ�A�* 

Average reward per step���'?N       ��2	O@�ǩ�A�*

epsilon����#�=.       ��W�	<M��ǩ�A�* 

Average reward per step����E�g       ��2	MN��ǩ�A�*

epsilon�����:�.       ��W�	�i��ǩ�A�* 

Average reward per step�������       ��2	@j��ǩ�A�*

epsilon���^5.�.       ��W�	����ǩ�A�* 

Average reward per step���˦B�       ��2	����ǩ�A�*

epsilon����v��.       ��W�	Bx��ǩ�A�* 

Average reward per step�����BY       ��2	�x��ǩ�A�*

epsilon���ӽ��.       ��W�	it��ǩ�A�* 

Average reward per step����/�       ��2	.u��ǩ�A�*

epsilon����ꄜ.       ��W�	̘��ǩ�A�* 

Average reward per step���0�]        ��2	k���ǩ�A�*

epsilon����ᱶ.       ��W�	%���ǩ�A�* 

Average reward per step�������       ��2	ĳ��ǩ�A�*

epsilon������s.       ��W�	���ǩ�A�* 

Average reward per step���?|�8       ��2	ڮ��ǩ�A�*

epsilon���V�)�.       ��W�	L�ǩ�A�* 

Average reward per step���7�-       ��2	Ñ�ǩ�A�*

epsilon���LU.       ��W�	���ǩ�A�* 

Average reward per step���a��       ��2	���ǩ�A�*

epsilon���y��M.       ��W�	`��ǩ�A�* 

Average reward per step���B�8       ��2	���ǩ�A�*

epsilon���S�:�.       ��W�	b��ǩ�A�* 

Average reward per step����'��       ��2	���ǩ�A�*

epsilon����Ht.       ��W�	7���ǩ�A�* 

Average reward per step����d��       ��2	 ��ǩ�A�*

epsilon����w �.       ��W�	>��ǩ�A�* 

Average reward per step����@�u       ��2	l��ǩ�A�*

epsilon����H	.       ��W�	)^��ǩ�A�* 

Average reward per step���f0g�       ��2	�^��ǩ�A�*

epsilon�����k.       ��W�	й��ǩ�A�* 

Average reward per step���N%�       ��2	����ǩ�A�*

epsilon����Q��.       ��W�	ϻ��ǩ�A�* 

Average reward per step����%8`       ��2	轠�ǩ�A�*

epsilon����9��.       ��W�	^���ǩ�A�* 

Average reward per step���Zr�j       ��2	@���ǩ�A�*

epsilon����u}.       ��W�	n¤�ǩ�A�* 

Average reward per step���f��U       ��2		ä�ǩ�A�*

epsilon����Ʃ�.       ��W�	���ǩ�A�* 

Average reward per step�����5I       ��2	\��ǩ�A�*

epsilon���8�.       ��W�	��ǩ�A�* 

Average reward per step���HΗ�       ��2	���ǩ�A�*

epsilon���r���.       ��W�	 	��ǩ�A�* 

Average reward per step���P���       ��2	�	��ǩ�A�*

epsilon����K$B.       ��W�	���ǩ�A�* 

Average reward per step���d���       ��2	� ��ǩ�A�*

epsilon���ǝ�.       ��W�	,*��ǩ�A�* 

Average reward per step������       ��2	�*��ǩ�A�*

epsilon���uP�.       ��W�	���ǩ�A�* 

Average reward per step���Fg��       ��2	\��ǩ�A�*

epsilon���#�\z.       ��W�	0��ǩ�A�* 

Average reward per step������       ��2	�0��ǩ�A�*

epsilon����#)l.       ��W�	�c��ǩ�A�* 

Average reward per step���PΕ�       ��2	�d��ǩ�A�*

epsilon�������.       ��W�	�m��ǩ�A�* 

Average reward per step������       ��2	in��ǩ�A�*

epsilon���.;.       ��W�	Sx��ǩ�A�* 

Average reward per step����Ժ`       ��2	%y��ǩ�A�*

epsilon�����.       ��W�	���ǩ�A�* 

Average reward per step���^�4       ��2	����ǩ�A�*

epsilon���ȴŞ.       ��W�	����ǩ�A�* 

Average reward per step���$�_�       ��2	T���ǩ�A�*

epsilon����|��.       ��W�	����ǩ�A�* 

Average reward per step�����%       ��2	n���ǩ�A�*

epsilon���1>4.       ��W�	�]��ǩ�A�* 

Average reward per step�����h�       ��2	�^��ǩ�A�*

epsilon���wv��0       ���_	����ǩ�A*#
!
Average reward per episodexx8���X.       ��W�	���ǩ�A*!

total reward per episode  Dµ�il.       ��W�	���ǩ�A�* 

Average reward per stepxx8��(o�       ��2	���ǩ�A�*

epsilonxx8�n�Hq.       ��W�	����ǩ�A�* 

Average reward per stepxx8�8f��       ��2	a���ǩ�A�*

epsilonxx8��6��.       ��W�	����ǩ�A�* 

Average reward per stepxx8��y��       ��2	Z���ǩ�A�*

epsilonxx8�6�.       ��W�	���ǩ�A�* 

Average reward per stepxx8����       ��2	����ǩ�A�*

epsilonxx8�wUs�.       ��W�	����ǩ�A�* 

Average reward per stepxx8�D:A       ��2	"���ǩ�A�*

epsilonxx8����.       ��W�	g���ǩ�A�* 

Average reward per stepxx8���@K       ��2	9���ǩ�A�*

epsilonxx8����{.       ��W�	���ǩ�A�* 

Average reward per stepxx8��$�       ��2	}���ǩ�A�*

epsilonxx8�)}�.       ��W�	����ǩ�A�* 

Average reward per stepxx8���*�       ��2	F���ǩ�A�*

epsilonxx8�28oi.       ��W�	���ǩ�A�* 

Average reward per stepxx8��V3x       ��2	����ǩ�A�*

epsilonxx8�ʤ13.       ��W�	���ǩ�A�* 

Average reward per stepxx8��#       ��2	����ǩ�A�*

epsilonxx8��u��.       ��W�	���ǩ�A�* 

Average reward per stepxx8���       ��2	����ǩ�A�*

epsilonxx8�v��.       ��W�	8���ǩ�A�* 

Average reward per stepxx8��<o       ��2	���ǩ�A�*

epsilonxx8�[�#@.       ��W�	W���ǩ�A�* 

Average reward per stepxx8�]��       ��2	���ǩ�A�*

epsilonxx8�yX#k.       ��W�	0��ǩ�A�* 

Average reward per stepxx8����       ��2	���ǩ�A�*

epsilonxx8��ϫG0       ���_	^+��ǩ�A*#
!
Average reward per episode%I:����.       ��W�	,��ǩ�A*!

total reward per episode  #�bKv.       ��W�	*8��ǩ�A�* 

Average reward per step%I:�Ľ��       ��2	�8��ǩ�A�*

epsilon%I:�J.       ��W�	YR��ǩ�A�* 

Average reward per step%I:�)���       ��2	*S��ǩ�A�*

epsilon%I:�}��.       ��W�	�T��ǩ�A�* 

Average reward per step%I:��w�-       ��2	"U��ǩ�A�*

epsilon%I:�e�BK.       ��W�	`X��ǩ�A�* 

Average reward per step%I:�]��_       ��2	�X��ǩ�A�*

epsilon%I:�ҿ4g.       ��W�	CW��ǩ�A�* 

Average reward per step%I:��g       ��2	X��ǩ�A�*

epsilon%I:�!C�.       ��W�	uY��ǩ�A�* 

Average reward per step%I:��UC�       ��2	Z��ǩ�A�*

epsilon%I:�n"�c.       ��W�	$`��ǩ�A�* 

Average reward per step%I:���M0       ��2	�`��ǩ�A�*

epsilon%I:����.       ��W�	�`��ǩ�A�* 

Average reward per step%I:�
�W�       ��2	�a��ǩ�A�*

epsilon%I:�.       ��W�	(e��ǩ�A�* 

Average reward per step%I:�v���       ��2	�e��ǩ�A�*

epsilon%I:�f�81.       ��W�	we��ǩ�A�* 

Average reward per step%I:�Ev       ��2	If��ǩ�A�*

epsilon%I:���t.       ��W�	m��ǩ�A�* 

Average reward per step%I:��-       ��2	n��ǩ�A�*

epsilon%I:�> 9x.       ��W�	"p��ǩ�A�* 

Average reward per step%I:���Ζ       ��2	�p��ǩ�A�*

epsilon%I:�}�	�.       ��W�	U���ǩ�A�* 

Average reward per step%I:���d       ��2	'���ǩ�A�*

epsilon%I:�t��.       ��W�	|���ǩ�A�* 

Average reward per step%I:��m�B       ��2	���ǩ�A�*

epsilon%I:���.       ��W�	˿��ǩ�A�* 

Average reward per step%I:��K       ��2	����ǩ�A�*

epsilon%I:��/�.       ��W�	���ǩ�A�* 

Average reward per step%I:�B�       ��2	���ǩ�A�*

epsilon%I:����.       ��W�	���ǩ�A�* 

Average reward per step%I:�a'��       ��2	4��ǩ�A�*

epsilon%I:��6}.       ��W�	���ǩ�A�* 

Average reward per step%I:�p��       ��2	8��ǩ�A�*

epsilon%I:��FQ�.       ��W�	���ǩ�A�* 

Average reward per step%I:��>��       ��2	���ǩ�A�*

epsilon%I:�(-��.       ��W�	}
�ǩ�A�* 

Average reward per step%I:���60       ��2	 
�ǩ�A�*

epsilon%I:��ڸ�.       ��W�	`��ǩ�A�* 

Average reward per step%I:��Ԟ�       ��2	>��ǩ�A�*

epsilon%I:�Kt��.       ��W�	��ǩ�A�* 

Average reward per step%I:��8�       ��2	���ǩ�A�*

epsilon%I:�SK*.       ��W�	��ǩ�A�* 

Average reward per step%I:�.3�       ��2	��ǩ�A�*

epsilon%I:�z<4�.       ��W�	�#�ǩ�A�* 

Average reward per step%I:����7       ��2	�$�ǩ�A�*

epsilon%I:���ʈ.       ��W�	��ǩ�A�* 

Average reward per step%I:�#(h       ��2	� �ǩ�A�*

epsilon%I:���ʋ.       ��W�	�B�ǩ�A�* 

Average reward per step%I:� %��       ��2	5C�ǩ�A�*

epsilon%I:�On�.       ��W�	�n�ǩ�A�* 

Average reward per step%I:�=�Oz       ��2	�o�ǩ�A�*

epsilon%I:���.       ��W�	���ǩ�A�* 

Average reward per step%I:�� ��       ��2	"��ǩ�A�*

epsilon%I:���j�.       ��W�	���ǩ�A�* 

Average reward per step%I:�۩�@       ��2	���ǩ�A�*

epsilon%I:�pX.       ��W�	V��ǩ�A�* 

Average reward per step%I:�~��!       ��2	��ǩ�A�*

epsilon%I:���.       ��W�	n��ǩ�A�* 

Average reward per step%I:�(�       ��2	+��ǩ�A�*

epsilon%I:��< �.       ��W�	�	!�ǩ�A�* 

Average reward per step%I:��--       ��2	�
!�ǩ�A�*

epsilon%I:���E.       ��W�	}y"�ǩ�A�* 

Average reward per step%I:��M�       ��2	 z"�ǩ�A�*

epsilon%I:���%X.       ��W�	�$�ǩ�A�* 

Average reward per step%I:���K5       ��2	��$�ǩ�A�*

epsilon%I:����.       ��W�	��&�ǩ�A�* 

Average reward per step%I:��1��       ��2	��&�ǩ�A�*

epsilon%I:��+�q.       ��W�	7(�ǩ�A�* 

Average reward per step%I:����d       ��2	�7(�ǩ�A�*

epsilon%I:��
�0       ���_	�O(�ǩ�A*#
!
Average reward per episodeUUe�[M|.       ��W�	P(�ǩ�A*!

total reward per episode  �o,Q�.       ��W�	�T,�ǩ�A�* 

Average reward per stepUUe�����       ��2	\U,�ǩ�A�*

epsilonUUe�u��g.       ��W�	�b.�ǩ�A�* 

Average reward per stepUUe���#S       ��2	9c.�ǩ�A�*

epsilonUUe�0�b .       ��W�	�V0�ǩ�A�* 

Average reward per stepUUe���c5       ��2	!W0�ǩ�A�*

epsilonUUe�WF�.       ��W�	��1�ǩ�A�* 

Average reward per stepUUe��"^       ��2	\�1�ǩ�A�*

epsilonUUe�F�.       ��W�	q�3�ǩ�A�* 

Average reward per stepUUe���
�       ��2	�3�ǩ�A�*

epsilonUUe�aSk�.       ��W�	�5�ǩ�A�* 

Average reward per stepUUe�g�D       ��2	��5�ǩ�A�*

epsilonUUe��C'.       ��W�	��7�ǩ�A�* 

Average reward per stepUUe���U�       ��2	T�7�ǩ�A�*

epsilonUUe�����.       ��W�	��9�ǩ�A�* 

Average reward per stepUUe��:�       ��2	!�9�ǩ�A�*

epsilonUUe�	��.       ��W�	��;�ǩ�A�* 

Average reward per stepUUe�7���       ��2	��;�ǩ�A�*

epsilonUUe����[.       ��W�	+P=�ǩ�A�* 

Average reward per stepUUe�����       ��2	Q=�ǩ�A�*

epsilonUUe���o.       ��W�	�v?�ǩ�A�* 

Average reward per stepUUe�H�'F       ��2	�w?�ǩ�A�*

epsilonUUe����.       ��W�	��A�ǩ�A�* 

Average reward per stepUUe�ĥ
       ��2	��A�ǩ�A�*

epsilonUUe��!�.       ��W�	ƨC�ǩ�A�* 

Average reward per stepUUe�-u�       ��2	��C�ǩ�A�*

epsilonUUe��l_.       ��W�	j�E�ǩ�A�* 

Average reward per stepUUe����       ��2	Q�E�ǩ�A�*

epsilonUUe�*@g�0       ���_	��E�ǩ�A*#
!
Average reward per episode%I:��~v.       ��W�	E�E�ǩ�A*!

total reward per episode  #Áq.       ��W�	�tI�ǩ�A�* 

Average reward per step%I:��P��       ��2	\uI�ǩ�A�*

epsilon%I:�~F�.       ��W�	V�K�ǩ�A�* 

Average reward per step%I:�-���       ��2	�K�ǩ�A�*

epsilon%I:����.       ��W�	��M�ǩ�A�* 

Average reward per step%I:�a���       ��2	}�M�ǩ�A�*

epsilon%I:�C)��.       ��W�	��O�ǩ�A�* 

Average reward per step%I:��+�       ��2	��O�ǩ�A�*

epsilon%I:�kz�.       ��W�	RR�ǩ�A�* 

Average reward per step%I:��+��       ��2	R�ǩ�A�*

epsilon%I:�a��%.       ��W�	5$T�ǩ�A�* 

Average reward per step%I:�Wz�       ��2	�$T�ǩ�A�*

epsilon%I:�<(�$.       ��W�	.XV�ǩ�A�* 

Average reward per step%I:��\/W       ��2	�XV�ǩ�A�*

epsilon%I:�;{�.       ��W�	>�W�ǩ�A�* 

Average reward per step%I:���-�       ��2	%�W�ǩ�A�*

epsilon%I:��9j�.       ��W�	�Z�ǩ�A�* 

Average reward per step%I:���       ��2	�Z�ǩ�A�*

epsilon%I:�)z�>.       ��W�	��[�ǩ�A�* 

Average reward per step%I:��)�       ��2	a�[�ǩ�A�*

epsilon%I:��a<�.       ��W�	��]�ǩ�A�* 

Average reward per step%I:�镾�       ��2	\�]�ǩ�A�*

epsilon%I:�0#?.       ��W�	�`�ǩ�A�* 

Average reward per step%I:�(�vP       ��2	�`�ǩ�A�*

epsilon%I:�����.       ��W�	�Hb�ǩ�A�* 

Average reward per step%I:���qx       ��2	(Ib�ǩ�A�*

epsilon%I:�5 .       ��W�	sLd�ǩ�A�* 

Average reward per step%I:���(z       ��2	IMd�ǩ�A�*

epsilon%I:��8�.       ��W�	�_f�ǩ�A�* 

Average reward per step%I:����       ��2	�`f�ǩ�A�*

epsilon%I:�gO�.       ��W�	P�h�ǩ�A�* 

Average reward per step%I:�\<�7       ��2	�h�ǩ�A�*

epsilon%I:���c#.       ��W�	z�j�ǩ�A�* 

Average reward per step%I:�%�Z       ��2	C�j�ǩ�A�*

epsilon%I:� �7.       ��W�	F�l�ǩ�A�* 

Average reward per step%I:����       ��2	(�l�ǩ�A�*

epsilon%I:���.       ��W�	��n�ǩ�A�* 

Average reward per step%I:���ty       ��2	Q�n�ǩ�A�*

epsilon%I:����.       ��W�	�3p�ǩ�A�* 

Average reward per step%I:�~荒       ��2	�4p�ǩ�A�*

epsilon%I:���x�.       ��W�	`Wr�ǩ�A�* 

Average reward per step%I:���ֆ       ��2	2Xr�ǩ�A�*

epsilon%I:�u	{3.       ��W�	�ot�ǩ�A�* 

Average reward per step%I:�)�Ⱥ       ��2	Xpt�ǩ�A�*

epsilon%I:�����.       ��W�	v�ǩ�A�* 

Average reward per step%I:�4�U       ��2	�v�ǩ�A�*

epsilon%I:�wc�K.       ��W�	M�x�ǩ�A�* 

Average reward per step%I:��F�       ��2	#�x�ǩ�A�*

epsilon%I:��,+.       ��W�	�z�ǩ�A�* 

Average reward per step%I:����       ��2	��z�ǩ�A�*

epsilon%I:���2.       ��W�	!�|�ǩ�A�* 

Average reward per step%I:�
��       ��2	�|�ǩ�A�*

epsilon%I:�y��f.       ��W�	|�~�ǩ�A�* 

Average reward per step%I:��ps�       ��2	J�~�ǩ�A�*

epsilon%I:�y6�.       ��W�	���ǩ�A�* 

Average reward per step%I:��6+(       ��2	����ǩ�A�*

epsilon%I:���d.       ��W�	g&��ǩ�A�* 

Average reward per step%I:�[3�       ��2	'��ǩ�A�*

epsilon%I:��<.       ��W�	@3��ǩ�A�* 

Average reward per step%I:�g�*Q       ��2	�3��ǩ�A�*

epsilon%I:�Jc�.       ��W�	H��ǩ�A�* 

Average reward per step%I:��t$q       ��2	�H��ǩ�A�*

epsilon%I:�~�"0       ���_	�j��ǩ�A*#
!
Average reward per episode�s��H�>.       ��W�	nk��ǩ�A*!

total reward per episode  
Û1�.       ��W�	
��ǩ�A�* 

Average reward per step�s�����L       ��2	�
��ǩ�A�*

epsilon�s�����.       ��W�	��ǩ�A�* 

Average reward per step�s��xռ�       ��2	���ǩ�A�*

epsilon�s���Γ.       ��W�	 ��ǩ�A�* 

Average reward per step�s���F�       ��2	� ��ǩ�A�*

epsilon�s���"y�.       ��W�	�3��ǩ�A�* 

Average reward per step�s���0�       ��2	D4��ǩ�A�*

epsilon�s��e���.       ��W�	�>��ǩ�A�* 

Average reward per step�s��p�       ��2	�?��ǩ�A�*

epsilon�s���cu%.       ��W�	rO��ǩ�A�* 

Average reward per step�s���v       ��2	P��ǩ�A�*

epsilon�s���.       ��W�	Zh��ǩ�A�* 

Average reward per step�s������       ��2	�h��ǩ�A�*

epsilon�s����i�.       ��W�	`x��ǩ�A�* 

Average reward per step�s���n"�       ��2	�x��ǩ�A�*

epsilon�s���֕x.       ��W�	G���ǩ�A�* 

Average reward per step�s������       ��2	���ǩ�A�*

epsilon�s��fE².       ��W�	陝�ǩ�A�* 

Average reward per step�s��̾^       ��2	����ǩ�A�*

epsilon�s�����.       ��W�	����ǩ�A�* 

Average reward per step�s����s�       ��2	����ǩ�A�*

epsilon�s���).       ��W�	����ǩ�A�* 

Average reward per step�s���D>       ��2	(���ǩ�A�*

epsilon�s���c0       ���_	����ǩ�A*#
!
Average reward per episode��f� ��f.       ��W�	���ǩ�A*!

total reward per episode  -�&��/.       ��W�	!��ǩ�A�* 

Average reward per step��f�J�       ��2	�!��ǩ�A�*

epsilon��f�u.       ��W�	Z��ǩ�A�* 

Average reward per step��f�g�       ��2	A���ǩ�A�*

epsilon��f�I �.       ��W�	#���ǩ�A�* 

Average reward per step��f�G���       ��2	����ǩ�A�*

epsilon��f�w8b.       ��W�	I���ǩ�A�* 

Average reward per step��f�r�D�       ��2	轪�ǩ�A�*

epsilon��f��}G�.       ��W�	qˬ�ǩ�A�* 

Average reward per step��f��iH       ��2	2̬�ǩ�A�*

epsilon��f�E�b.       ��W�	��ǩ�A�* 

Average reward per step��f�i��/       ��2	���ǩ�A�*

epsilon��f��o��.       ��W�	����ǩ�A�* 

Average reward per step��f�e�       ��2	����ǩ�A�*

epsilon��f�M��..       ��W�	t��ǩ�A�* 

Average reward per step��f���`       ��2	��ǩ�A�*

epsilon��f��,.5.       ��W�	��ǩ�A�* 

Average reward per step��f��X�&       ��2	���ǩ�A�*

epsilon��f�߂�.       ��W�	.9��ǩ�A�* 

Average reward per step��f���x       ��2	�9��ǩ�A�*

epsilon��f���.       ��W�	����ǩ�A�* 

Average reward per step��f��a)\       ��2	5���ǩ�A�*

epsilon��f�}�<�.       ��W�	Cƺ�ǩ�A�* 

Average reward per step��f�S��       ��2	XǺ�ǩ�A�*

epsilon��f�YA<�.       ��W�	���ǩ�A�* 

Average reward per step��f��4$       ��2	���ǩ�A�*

epsilon��f�2�N.       ��W�	���ǩ�A�* 

Average reward per step��f�l��       ��2	A���ǩ�A�*

epsilon��f��B��.       ��W�	��ǩ�A�* 

Average reward per step��f�$�;�       ��2	���ǩ�A�*

epsilon��f�/�,�.       ��W�	I��ǩ�A�* 

Average reward per step��f�h�>�       ��2	,��ǩ�A�*

epsilon��f�����.       ��W�	�!��ǩ�A�* 

Average reward per step��f��#j�       ��2	l"��ǩ�A�*

epsilon��f�)��3.       ��W�	�.��ǩ�A�* 

Average reward per step��f��[�!       ��2	�/��ǩ�A�*

epsilon��f��(@.       ��W�	tF��ǩ�A�* 

Average reward per step��f�`�x       ��2	RG��ǩ�A�*

epsilon��f����.       ��W�	F���ǩ�A�* 

Average reward per step��f�X9��       ��2	V���ǩ�A�*

epsilon��f���jY.       ��W�	;��ǩ�A�* 

Average reward per step��f���e\       ��2	���ǩ�A�*

epsilon��f�~p�.       ��W�	�5��ǩ�A�* 

Average reward per step��f�/b�       ��2	i6��ǩ�A�*

epsilon��f�V��.       ��W�	�0��ǩ�A�* 

Average reward per step��f�o��]       ��2	�1��ǩ�A�*

epsilon��f��x�0       ���_	�Q��ǩ�A*#
!
Average reward per episode-d��Ek�.       ��W�	/R��ǩ�A*!

total reward per episode  "�*=u�.       ��W�	����ǩ�A�* 

Average reward per step-d����       ��2	]���ǩ�A�*

epsilon-d��4�F}.       ��W�	���ǩ�A�* 

Average reward per step-d��ͻ��       ��2	{���ǩ�A�*

epsilon-d��kY�.       ��W�		q��ǩ�A�* 

Average reward per step-d��)j�)       ��2	�q��ǩ�A�*

epsilon-d��@�g.       ��W�	9���ǩ�A�* 

Average reward per step-d�����{       ��2	Ԛ��ǩ�A�*

epsilon-d��n�Y.       ��W�	����ǩ�A�* 

Average reward per step-d��g��"       ��2	~���ǩ�A�*

epsilon-d���^h�.       ��W�	3���ǩ�A�* 

Average reward per step-d��%���       ��2	����ǩ�A�*

epsilon-d��espr.       ��W�	�U��ǩ�A�* 

Average reward per step-d��Z�       ��2	\V��ǩ�A�*

epsilon-d��,��.       ��W�	3o��ǩ�A�* 

Average reward per step-d���U�       ��2	�o��ǩ�A�*

epsilon-d����ٳ.       ��W�	����ǩ�A�* 

Average reward per step-d��)�9�       ��2	����ǩ�A�*

epsilon-d��Ɯ.       ��W�	����ǩ�A�* 

Average reward per step-d�����S       ��2	����ǩ�A�*

epsilon-d��X^�.       ��W�	����ǩ�A�* 

Average reward per step-d����       ��2	����ǩ�A�*

epsilon-d���бU.       ��W�	���ǩ�A�* 

Average reward per step-d��>�i       ��2	����ǩ�A�*

epsilon-d���P��.       ��W�	��ǩ�A�* 

Average reward per step-d��d,�k       ��2	���ǩ�A�*

epsilon-d��Ǧ�.       ��W�	� ��ǩ�A�* 

Average reward per step-d���( �       ��2	S!��ǩ�A�*

epsilon-d��{dr.       ��W�	�#��ǩ�A�* 

Average reward per step-d��Á�       ��2	�$��ǩ�A�*

epsilon-d����.       ��W�	�I��ǩ�A�* 

Average reward per step-d��iً       ��2	AJ��ǩ�A�*

epsilon-d��c��.       ��W�	eS��ǩ�A�* 

Average reward per step-d��@�L       ��2	T��ǩ�A�*

epsilon-d��@RF.       ��W�	_��ǩ�A�* 

Average reward per step-d��	Q[       ��2	�_��ǩ�A�*

epsilon-d���6n.       ��W�	Ag��ǩ�A�* 

Average reward per step-d����s�       ��2	�g��ǩ�A�*

epsilon-d��o�U.       ��W�	,���ǩ�A�* 

Average reward per step-d�� �v       ��2	���ǩ�A�*

epsilon-d��lQ.       ��W�	���ǩ�A�* 

Average reward per step-d��ㅲ�       ��2	r��ǩ�A�*

epsilon-d������.       ��W�	8��ǩ�A�* 

Average reward per step-d��x��}       ��2	�8��ǩ�A�*

epsilon-d��ͨ�.       ��W�	JD�ǩ�A�* 

Average reward per step-d��3o��       ��2	E�ǩ�A�*

epsilon-d��ʞ�.       ��W�	a�ǩ�A�* 

Average reward per step-d��	:�$       ��2	�a�ǩ�A�*

epsilon-d��:.y�.       ��W�	V�ǩ�A�* 

Average reward per step-d����P       ��2	�V�ǩ�A�*

epsilon-d����4�.       ��W�	�c�ǩ�A�* 

Average reward per step-d��;�Fq       ��2	�d�ǩ�A�*

epsilon-d����v�.       ��W�	�v	�ǩ�A�* 

Average reward per step-d��B��       ��2	Ww	�ǩ�A�*

epsilon-d���|na.       ��W�	���ǩ�A�* 

Average reward per step-d��*6�6       ��2	W��ǩ�A�*

epsilon-d����U.       ��W�	%��ǩ�A�* 

Average reward per step-d���?�       ��2	��ǩ�A�*

epsilon-d���k��.       ��W�	���ǩ�A�* 

Average reward per step-d������       ��2	]��ǩ�A�*

epsilon-d��7�o�.       ��W�	���ǩ�A�* 

Average reward per step-d���$�       ��2	t��ǩ�A�*

epsilon-d���[�.       ��W�	���ǩ�A�* 

Average reward per step-d����KP       ��2	���ǩ�A�*

epsilon-d������.       ��W�	f��ǩ�A�* 

Average reward per step-d��*E��       ��2		��ǩ�A�*

epsilon-d���n�1.       ��W�	�ǩ�A�* 

Average reward per step-d��KsS8       ��2	��ǩ�A�*

epsilon-d����.       ��W�	dy�ǩ�A�* 

Average reward per step-d���T�       ��2	)z�ǩ�A�*

epsilon-d��#I^�.       ��W�	O��ǩ�A�* 

Average reward per step-d���,�       ��2	:��ǩ�A�*

epsilon-d��{�p�.       ��W�	�	�ǩ�A�* 

Average reward per step-d��PUʉ       ��2	�
�ǩ�A�*

epsilon-d��..       ��W�	p%�ǩ�A�* 

Average reward per step-d���dH9       ��2	&�ǩ�A�*

epsilon-d��1�q.       ��W�	��!�ǩ�A�* 

Average reward per step-d��c�L       ��2	��!�ǩ�A�*

epsilon-d���l�.       ��W�	T6%�ǩ�A�* 

Average reward per step-d��\�D�       ��2	�6%�ǩ�A�*

epsilon-d��@'�g.       ��W�	fN'�ǩ�A�* 

Average reward per step-d���0q�       ��2	"O'�ǩ�A�*

epsilon-d��q�.       ��W�	��(�ǩ�A�* 

Average reward per step-d��q��       ��2	��(�ǩ�A�*

epsilon-d����.       ��W�	�7*�ǩ�A�* 

Average reward per step-d��%}��       ��2	C8*�ǩ�A�*

epsilon-d���l�;.       ��W�	jh,�ǩ�A�* 

Average reward per step-d���b�       ��2	i,�ǩ�A�*

epsilon-d���1E.       ��W�	k�.�ǩ�A�* 

Average reward per step-d���Z�E       ��2	E�.�ǩ�A�*

epsilon-d�����,.       ��W�	Ԛ0�ǩ�A�* 

Average reward per step-d������       ��2	x�0�ǩ�A�*

epsilon-d��_��.       ��W�	�2�ǩ�A�* 

Average reward per step-d���9       ��2	S2�ǩ�A�*

epsilon-d��?�.       ��W�	$a3�ǩ�A�* 

Average reward per step-d�����       ��2	b3�ǩ�A�*

epsilon-d��l��\.       ��W�	��5�ǩ�A�* 

Average reward per step-d���z�s       ��2	E�5�ǩ�A�*

epsilon-d��|�Th.       ��W�	�,7�ǩ�A�* 

Average reward per step-d�����       ��2	s-7�ǩ�A�*

epsilon-d��h� �.       ��W�	�89�ǩ�A�* 

Average reward per step-d��KH��       ��2	e99�ǩ�A�*

epsilon-d�����.       ��W�	1�;�ǩ�A�* 

Average reward per step-d��|�9       ��2	��;�ǩ�A�*

epsilon-d����Zc.       ��W�	>Y=�ǩ�A�* 

Average reward per step-d��r�!t       ��2	�Y=�ǩ�A�*

epsilon-d��tu .       ��W�	�x?�ǩ�A�* 

Average reward per step-d��!�       ��2	�y?�ǩ�A�*

epsilon-d��>ri.       ��W�	��A�ǩ�A�* 

Average reward per step-d��ˆN       ��2	X�A�ǩ�A�*

epsilon-d��?��|.       ��W�	��C�ǩ�A�* 

Average reward per step-d���Y��       ��2	R�C�ǩ�A�*

epsilon-d��*��@.       ��W�	��E�ǩ�A�* 

Average reward per step-d���q�       ��2	��E�ǩ�A�*

epsilon-d���=.       ��W�	=�G�ǩ�A�* 

Average reward per step-d��o���       ��2	��G�ǩ�A�*

epsilon-d�����s.       ��W�	��I�ǩ�A�* 

Average reward per step-d������       ��2	��I�ǩ�A�*

epsilon-d���ԓ�.       ��W�	L�ǩ�A�* 

Average reward per step-d���n#       ��2	�L�ǩ�A�*

epsilon-d���U�u.       ��W�	$(N�ǩ�A�* 

Average reward per step-d��H�&1       ��2	�(N�ǩ�A�*

epsilon-d���縅.       ��W�	EJP�ǩ�A�* 

Average reward per step-d��c�$H       ��2	�JP�ǩ�A�*

epsilon-d��>�.       ��W�	cFR�ǩ�A�* 

Average reward per step-d��~"�o       ��2	�FR�ǩ�A�*

epsilon-d��L��.       ��W�	lZT�ǩ�A�* 

Average reward per step-d��ECy�       ��2	-[T�ǩ�A�*

epsilon-d����=.       ��W�		lV�ǩ�A�* 

Average reward per step-d����`       ��2	�lV�ǩ�A�*

epsilon-d����$0       ���_	ߊV�ǩ�A *#
!
Average reward per episode�Nl��_L.       ��W�	T�V�ǩ�A *!

total reward per episode  pr5.       ��W�	�Z�ǩ�A�* 

Average reward per step�Nl�Î&(       ��2	dZ�ǩ�A�*

epsilon�Nl�:%��.       ��W�	\�ǩ�A�* 

Average reward per step�Nl�C̤�       ��2	�\�ǩ�A�*

epsilon�Nl��;�.       ��W�	�#^�ǩ�A�* 

Average reward per step�Nl����z       ��2	F$^�ǩ�A�*

epsilon�Nl�t�ɸ.       ��W�	�?`�ǩ�A�* 

Average reward per step�Nl�1:�       ��2	_@`�ǩ�A�*

epsilon�Nl����.       ��W�	�Zb�ǩ�A�* 

Average reward per step�Nl��
�I       ��2	O[b�ǩ�A�*

epsilon�Nl�aRH;.       ��W�	1_d�ǩ�A�* 

Average reward per step�Nl��H�5       ��2	�_d�ǩ�A�*

epsilon�Nl�%MNT.       ��W�	__f�ǩ�A�* 

Average reward per step�Nl�~H�       ��2	`f�ǩ�A�*

epsilon�Nl�����.       ��W�	~rh�ǩ�A�* 

Average reward per step�Nl��       ��2	Tsh�ǩ�A�*

epsilon�Nl���.       ��W�	�j�ǩ�A�* 

Average reward per step�Nl�U�]�       ��2	��j�ǩ�A�*

epsilon�Nl�we��.       ��W�	�l�ǩ�A�* 

Average reward per step�Nl��ߺq       ��2	��l�ǩ�A�*

epsilon�Nl��9.       ��W�	\�n�ǩ�A�* 

Average reward per step�Nl�,���       ��2	�n�ǩ�A�*

epsilon�Nl���.       ��W�	Ūp�ǩ�A�* 

Average reward per step�Nl���       ��2	��p�ǩ�A�*

epsilon�Nl���.       ��W�	�r�ǩ�A�* 

Average reward per step�Nl�K���       ��2	nr�ǩ�A�*

epsilon�Nl���I.       ��W�	O t�ǩ�A�* 

Average reward per step�Nl��Z�>       ��2	� t�ǩ�A�*

epsilon�Nl�G�h�0       ���_	9t�ǩ�A!*#
!
Average reward per episoden�>�_�==.       ��W�	�9t�ǩ�A!*!

total reward per episode  '��0ɿ.       ��W�	�;w�ǩ�A�* 

Average reward per stepn�>��SW�       ��2	�<w�ǩ�A�*

epsilonn�>�o1e
.       ��W�	8hy�ǩ�A�* 

Average reward per stepn�>�ƀ�9       ��2	iy�ǩ�A�*

epsilonn�>��AA�.       ��W�	�{{�ǩ�A�* 

Average reward per stepn�>�~E`�       ��2	c|{�ǩ�A�*

epsilonn�>�g.��.       ��W�	��}�ǩ�A�* 

Average reward per stepn�>��zJ�       ��2	b�}�ǩ�A�*

epsilonn�>�8��	.       ��W�	ǁ�ǩ�A�* 

Average reward per stepn�>��z��       ��2	b��ǩ�A�*

epsilonn�>���Xe.       ��W�	Sˁ�ǩ�A�* 

Average reward per stepn�>��˛�       ��2	́�ǩ�A�*

epsilonn�>�&<P.       ��W�	^L��ǩ�A�* 

Average reward per stepn�>��6�E       ��2	#M��ǩ�A�*

epsilonn�>�%~�E.       ��W�	h��ǩ�A�* 

Average reward per stepn�>�W���       ��2	�h��ǩ�A�*

epsilonn�>�%�D.       ��W�	Gv��ǩ�A�* 

Average reward per stepn�>����       ��2	�v��ǩ�A�*

epsilonn�>����.       ��W�	�v��ǩ�A�* 

Average reward per stepn�>�[��Z       ��2	\w��ǩ�A�*

epsilonn�>�;��t.       ��W�	�s��ǩ�A�* 

Average reward per stepn�>�5�f�       ��2	�t��ǩ�A�*

epsilonn�>�^�G.       ��W�	Z���ǩ�A�* 

Average reward per stepn�>�BW��       ��2	����ǩ�A�*

epsilonn�>�.��~.       ��W�	���ǩ�A�* 

Average reward per stepn�>��O�       ��2	ʈ��ǩ�A�*

epsilonn�>��]�t.       ��W�	[���ǩ�A�* 

Average reward per stepn�>�Ҁ�       ��2	��ǩ�A�*

epsilonn�>�d�#�.       ��W�	`���ǩ�A�* 

Average reward per stepn�>�3��       ��2	2���ǩ�A�*

epsilonn�>��~�;.       ��W�	�˕�ǩ�A�	* 

Average reward per stepn�>��!       ��2	\̕�ǩ�A�	*

epsilonn�>�GT��.       ��W�	�ܗ�ǩ�A�	* 

Average reward per stepn�>�4��       ��2	aݗ�ǩ�A�	*

epsilonn�>�ms�d.       ��W�	����ǩ�A�	* 

Average reward per stepn�>����       ��2	Y���ǩ�A�	*

epsilonn�>���.       ��W�	���ǩ�A�	* 

Average reward per stepn�>�M�q�       ��2	���ǩ�A�	*

epsilonn�>�rQ�q.       ��W�	���ǩ�A�	* 

Average reward per stepn�>���>�       ��2	���ǩ�A�	*

epsilonn�>��$!H.       ��W�	���ǩ�A�	* 

Average reward per stepn�>�w�>�       ��2	���ǩ�A�	*

epsilonn�>����.       ��W�	?��ǩ�A�	* 

Average reward per stepn�>���&t       ��2	�?��ǩ�A�	*

epsilonn�>��s�T.       ��W�	�D��ǩ�A�	* 

Average reward per stepn�>����       ��2	�E��ǩ�A�	*

epsilonn�>�<�d$.       ��W�	�g��ǩ�A�	* 

Average reward per stepn�>�8H�.       ��2	Qh��ǩ�A�	*

epsilonn�>�N��1.       ��W�	Hn��ǩ�A�	* 

Average reward per stepn�>�,�\�       ��2	o��ǩ�A�	*

epsilonn�>�R���.       ��W�	Â��ǩ�A�	* 

Average reward per stepn�>�.�p$       ��2	����ǩ�A�	*

epsilonn�>�Ћ}:.       ��W�	����ǩ�A�	* 

Average reward per stepn�>���3�       ��2	d���ǩ�A�	*

epsilonn�>��>l�.       ��W�	V���ǩ�A�	* 

Average reward per stepn�>�	Cw�       ��2	鶮�ǩ�A�	*

epsilonn�>��>��.       ��W�	��ǩ�A�	* 

Average reward per stepn�>�)Ӄ�       ��2	���ǩ�A�	*

epsilonn�>����g.       ��W�	���ǩ�A�	* 

Average reward per stepn�>�׳u`       ��2	����ǩ�A�	*

epsilonn�>��]{�.       ��W�	e���ǩ�A�	* 

Average reward per stepn�>�*�9       ��2	?���ǩ�A�	*

epsilonn�>���.       ��W�	U���ǩ�A�	* 

Average reward per stepn�>��C��       ��2	迵�ǩ�A�	*

epsilonn�>��
��.       ��W�	����ǩ�A�	* 

Average reward per stepn�>�]8}       ��2	]���ǩ�A�	*

epsilonn�>���A0       ���_	Yܷ�ǩ�A"*#
!
Average reward per episodeN6Y�.J�K.       ��W�	�ܷ�ǩ�A"*!

total reward per episode  ��cV�.       ��W�	9���ǩ�A�	* 

Average reward per stepN6Y�\�f�       ��2	����ǩ�A�	*

epsilonN6Y�Z��O.       ��W�	�ؼ�ǩ�A�	* 

Average reward per stepN6Y�s	ǝ       ��2	�ټ�ǩ�A�	*

epsilonN6Y���5.       ��W�	����ǩ�A�	* 

Average reward per stepN6Y�����       ��2	����ǩ�A�	*

epsilonN6Y�ˑ�o.       ��W�	���ǩ�A�	* 

Average reward per stepN6Y�l�q       ��2	O��ǩ�A�	*

epsilonN6Y�����.       ��W�	���ǩ�A�	* 

Average reward per stepN6Y�7�|w       ��2	^��ǩ�A�	*

epsilonN6Y�����.       ��W�	X��ǩ�A�	* 

Average reward per stepN6Y��޸G       ��2	���ǩ�A�	*

epsilonN6Y�xۻ.       ��W�	�(��ǩ�A�	* 

Average reward per stepN6Y�6�6W       ��2	�)��ǩ�A�	*

epsilonN6Y��|X�.       ��W�	-��ǩ�A�	* 

Average reward per stepN6Y��� \       ��2	�-��ǩ�A�	*

epsilonN6Y���".       ��W�	�6��ǩ�A�	* 

Average reward per stepN6Y��׋/       ��2	37��ǩ�A�	*

epsilonN6Y���1�.       ��W�	�:��ǩ�A�	* 

Average reward per stepN6Y�eo~�       ��2	�;��ǩ�A�	*

epsilonN6Y�VR.�.       ��W�	�h��ǩ�A�	* 

Average reward per stepN6Y�R���       ��2	�i��ǩ�A�	*

epsilonN6Y�~�.       ��W�	F���ǩ�A�	* 

Average reward per stepN6Y��,��       ��2	����ǩ�A�	*

epsilonN6Y��p�.       ��W�	�1��ǩ�A�	* 

Average reward per stepN6Y�0�       ��2	32��ǩ�A�	*

epsilonN6Y�Tz(�.       ��W�	�A��ǩ�A�	* 

Average reward per stepN6Y�ջs       ��2	�B��ǩ�A�	*

epsilonN6Y�Gݧ2.       ��W�	�`��ǩ�A�	* 

Average reward per stepN6Y�El\       ��2	[a��ǩ�A�	*

epsilonN6Y��h;.       ��W�	�j��ǩ�A�	* 

Average reward per stepN6Y����       ��2	Yk��ǩ�A�	*

epsilonN6Y�����.       ��W�	���ǩ�A�	* 

Average reward per stepN6Y��$<8       ��2	̚��ǩ�A�	*

epsilonN6Y�M\�.       ��W�	���ǩ�A�	* 

Average reward per stepN6Y���OY       ��2	����ǩ�A�	*

epsilonN6Y�^���.       ��W�	����ǩ�A�	* 

Average reward per stepN6Y�;k4/       ��2	S���ǩ�A�	*

epsilonN6Y�o;>9.       ��W�	c*��ǩ�A�	* 

Average reward per stepN6Y�֡�w       ��2	0+��ǩ�A�	*

epsilonN6Y��c�u0       ���_	sH��ǩ�A#*#
!
Average reward per episode�����5�w.       ��W�	(I��ǩ�A#*!

total reward per episode  ����.       ��W�	�t��ǩ�A�	* 

Average reward per step�������       ��2	Xu��ǩ�A�	*

epsilon����+[�A.       ��W�	���ǩ�A�	* 

Average reward per step�����q��       ��2	����ǩ�A�	*

epsilon������N.       ��W�	����ǩ�A�	* 

Average reward per step�����Q�7       ��2	U���ǩ�A�	*

epsilon������.       ��W�	Q���ǩ�A�	* 

Average reward per step�����,q       ��2	
���ǩ�A�	*

epsilon�����"�.       ��W�	���ǩ�A�	* 

Average reward per step����U��       ��2	����ǩ�A�	*

epsilon����̈́�1.       ��W�	���ǩ�A�	* 

Average reward per step������V�       ��2	���ǩ�A�	*

epsilon����(��.       ��W�	�g��ǩ�A�	* 

Average reward per step�����"v�       ��2	�h��ǩ�A�	*

epsilon����K�k.       ��W�	A���ǩ�A�	* 

Average reward per step����V�W�       ��2	����ǩ�A�	*

epsilon�����M9f.       ��W�	����ǩ�A�	* 

Average reward per step����G�$�       ��2	B���ǩ�A�	*

epsilon����T4��.       ��W�	����ǩ�A�	* 

Average reward per step����$       ��2	����ǩ�A�	*

epsilon����Mͫ.       ��W�	N��ǩ�A�	* 

Average reward per step�����wb       ��2	(��ǩ�A�	*

epsilon�����!�.       ��W�	���ǩ�A�	* 

Average reward per step����K霁       ��2	} ��ǩ�A�	*

epsilon�����
��.       ��W�	�+��ǩ�A�	* 

Average reward per step����4�.1       ��2	A,��ǩ�A�	*

epsilon�����f��.       ��W�	v6��ǩ�A�	* 

Average reward per step����<؅       ��2	L7��ǩ�A�	*

epsilon����]��v.       ��W�	1B �ǩ�A�	* 

Average reward per step������r       ��2	�B �ǩ�A�	*

epsilon������3.       ��W�	N�ǩ�A�	* 

Average reward per step�������       ��2	�N�ǩ�A�	*

epsilon�����`4.       ��W�	]Q�ǩ�A�	* 

Average reward per step����ph՛       ��2	"R�ǩ�A�	*

epsilon��������.       ��W�	�W�ǩ�A�	* 

Average reward per step�����W��       ��2	qX�ǩ�A�	*

epsilon����A��).       ��W�	��ǩ�A�	* 

Average reward per step������65       ��2	���ǩ�A�	*

epsilon�����e�.       ��W�	��
�ǩ�A�	* 

Average reward per step�����t��       ��2	��
�ǩ�A�	*

epsilon����1V.       ��W�	 �ǩ�A�	* 

Average reward per step����UM(       ��2	� �ǩ�A�	*

epsilon�����cd<.       ��W�	�C�ǩ�A�	* 

Average reward per step����I���       ��2	JD�ǩ�A�	*

epsilon����s�.       ��W�	�L�ǩ�A�	* 

Average reward per step�������       ��2	�M�ǩ�A�	*

epsilon������.       ��W�	k��ǩ�A�	* 

Average reward per step����~��E       ��2	I��ǩ�A�	*

epsilon�����N��.       ��W�	��ǩ�A�	* 

Average reward per step����B�Ǖ       ��2	���ǩ�A�	*

epsilon������Ǘ.       ��W�	���ǩ�A�	* 

Average reward per step������       ��2	���ǩ�A�	*

epsilon�����)a.       ��W�	��ǩ�A�	* 

Average reward per step�����YE       ��2	���ǩ�A�	*

epsilon�����[�>.       ��W�	V��ǩ�A�	* 

Average reward per step����EkEZ       ��2	���ǩ�A�	*

epsilon�������.       ��W�	��ǩ�A�	* 

Average reward per step���� y>       ��2	��ǩ�A�	*

epsilon�����K!�.       ��W�	��ǩ�A�	* 

Average reward per step����D��q       ��2	/�ǩ�A�	*

epsilon������Q.       ��W�	9'!�ǩ�A�	* 

Average reward per step�������       ��2	�'!�ǩ�A�	*

epsilon����)��k.       ��W�	�1#�ǩ�A�	* 

Average reward per step������˫       ��2	Q2#�ǩ�A�	*

epsilon�����(0.       ��W�	9%�ǩ�A�	* 

Average reward per step�������8       ��2	�9%�ǩ�A�	*

epsilon����ד.       ��W�	[a'�ǩ�A�	* 

Average reward per step�����ե{       ��2	Rb'�ǩ�A�	*

epsilon�������.       ��W�	
�(�ǩ�A�	* 

Average reward per step�������       ��2	��(�ǩ�A�	*

epsilon����V�l.       ��W�	��*�ǩ�A�	* 

Average reward per step�����O��       ��2	8�*�ǩ�A�	*

epsilon�����"�.       ��W�	V-�ǩ�A�	* 

Average reward per step����/F�       ��2	$-�ǩ�A�	*

epsilon�������.       ��W�	�"/�ǩ�A�	* 

Average reward per step�������       ��2	�#/�ǩ�A�	*

epsilon�����*p�.       ��W�	�/1�ǩ�A�	* 

Average reward per step������{1       ��2	Y01�ǩ�A�	*

epsilon�����~N�.       ��W�	^J3�ǩ�A�	* 

Average reward per step�����       ��2	K3�ǩ�A�	*

epsilon�����N(E.       ��W�	~S5�ǩ�A�	* 

Average reward per step�����7�!       ��2	PT5�ǩ�A�	*

epsilon����Jw�.       ��W�	Wv7�ǩ�A�	* 

Average reward per step�����=�       ��2	�v7�ǩ�A�	*

epsilon����J#�.       ��W�	�9�ǩ�A�	* 

Average reward per step������O       ��2	��9�ǩ�A�	*

epsilon����N�.       ��W�	��;�ǩ�A�	* 

Average reward per step�����Ⱦ       ��2	֫;�ǩ�A�	*

epsilon������W�.       ��W�	:�=�ǩ�A�	* 

Average reward per step����OCb}       ��2	!�=�ǩ�A�	*

epsilon�����u%�.       ��W�	?�?�ǩ�A�	* 

Average reward per step����`8^       ��2	��?�ǩ�A�	*

epsilon����ay�4.       ��W�	�EA�ǩ�A�	* 

Average reward per step���� ި}       ��2	NFA�ǩ�A�	*

epsilon����
��.       ��W�	�gC�ǩ�A�	* 

Average reward per step����GL23       ��2	{hC�ǩ�A�	*

epsilon�����S��.       ��W�	ӤE�ǩ�A�	* 

Average reward per step����c�~y       ��2	��E�ǩ�A�	*

epsilon�����j��.       ��W�	"�G�ǩ�A�	* 

Average reward per step����R�       ��2	��G�ǩ�A�	*

epsilon�����+*�.       ��W�	C�I�ǩ�A�	* 

Average reward per step����<q�       ��2	�I�ǩ�A�	*

epsilon����ÆW�.       ��W�	ZL�ǩ�A�	* 

Average reward per step����pk��       ��2	�L�ǩ�A�	*

epsilon����⊂.       ��W�	�N�ǩ�A�	* 

Average reward per step����!K��       ��2	�N�ǩ�A�	*

epsilon������.       ��W�	#�O�ǩ�A�	* 

Average reward per step����v�       ��2	��O�ǩ�A�	*

epsilon����0��>.       ��W�	=�Q�ǩ�A�	* 

Average reward per step����&A�R       ��2	�Q�ǩ�A�	*

epsilon����
�8C.       ��W�	�T�ǩ�A�	* 

Average reward per step�����&B�       ��2	��T�ǩ�A�	*

epsilon������g�.       ��W�	��V�ǩ�A�	* 

Average reward per step������$q       ��2	ϣV�ǩ�A�	*

epsilon����,�a/.       ��W�	�X�ǩ�A�	* 

Average reward per step������A       ��2	�X�ǩ�A�	*

epsilon����Y\�.       ��W�	��Y�ǩ�A�	* 

Average reward per step����Rr�       ��2	P�Y�ǩ�A�	*

epsilon�����2|.       ��W�	ͯ[�ǩ�A�	* 

Average reward per step�������       ��2	y�[�ǩ�A�	*

epsilon�����^�h0       ���_	^�[�ǩ�A$*#
!
Average reward per episode""b�[`��.       ��W�	�[�ǩ�A$*!

total reward per episode  T �.       ��W�	1{_�ǩ�A�	* 

Average reward per step""b�OL�!       ��2	�{_�ǩ�A�	*

epsilon""b�A�O�.       ��W�	z�a�ǩ�A�	* 

Average reward per step""b����       ��2	P�a�ǩ�A�	*

epsilon""b����&.       ��W�	L�c�ǩ�A�	* 

Average reward per step""b�_��       ��2	"�c�ǩ�A�	*

epsilon""b��K>�.       ��W�	t�e�ǩ�A�	* 

Average reward per step""b���υ       ��2	�e�ǩ�A�	*

epsilon""b���iY.       ��W�	��g�ǩ�A�	* 

Average reward per step""b����       ��2	��g�ǩ�A�	*

epsilon""b��]�.       ��W�	
�i�ǩ�A�	* 

Average reward per step""b�F��       ��2	��i�ǩ�A�	*

epsilon""b�}��p.       ��W�	��k�ǩ�A�	* 

Average reward per step""b�ţ       ��2	��k�ǩ�A�	*

epsilon""b�Cߦ�.       ��W�		n�ǩ�A�	* 

Average reward per step""b�$��{       ��2	�	n�ǩ�A�	*

epsilon""b��xʤ.       ��W�	*p�ǩ�A�	* 

Average reward per step""b����       ��2	�*p�ǩ�A�	*

epsilon""b���U.       ��W�	�/r�ǩ�A�	* 

Average reward per step""b���O       ��2	�0r�ǩ�A�	*

epsilon""b�/�\.       ��W�	�Xt�ǩ�A�	* 

Average reward per step""b�$�i�       ��2	�Yt�ǩ�A�	*

epsilon""b�7˫.       ��W�	U�u�ǩ�A�	* 

Average reward per step""b��+       ��2	Y�u�ǩ�A�	*

epsilon""b��Kq�.       ��W�	|�w�ǩ�A�	* 

Average reward per step""b���       ��2	c�w�ǩ�A�	*

epsilon""b�Kk��.       ��W�	z�ǩ�A�	* 

Average reward per step""b��4N�       ��2	]z�ǩ�A�	*

epsilon""b���v�.       ��W�	T5|�ǩ�A�	* 

Average reward per step""b���Ç       ��2	/6|�ǩ�A�	*

epsilon""b����.       ��W�	DQ~�ǩ�A�	* 

Average reward per step""b�3S�@       ��2	�Q~�ǩ�A�	*

epsilon""b��П�.       ��W�	�z��ǩ�A�	* 

Average reward per step""b�\��1       ��2	|��ǩ�A�	*

epsilon""b�R�[{.       ��W�	����ǩ�A�	* 

Average reward per step""b��p�       ��2	᷂�ǩ�A�	*

epsilon""b�{��0       ���_	ۂ�ǩ�A%*#
!
Average reward per episode���t5.       ��W�	�ۂ�ǩ�A%*!

total reward per episode  #��S.       ��W�	�p��ǩ�A�	* 

Average reward per step�����        ��2	�q��ǩ�A�	*

epsilon���>��.       ��W�	���ǩ�A�	* 

Average reward per step����� #       ��2	����ǩ�A�	*

epsilon���*��`.       ��W�	=ӊ�ǩ�A�	* 

Average reward per step���6q�       ��2	�ӊ�ǩ�A�	*

epsilon���;X�.       ��W�	�=��ǩ�A�	* 

Average reward per step���b�u       ��2	6>��ǩ�A�	*

epsilon�����<k.       ��W�	{���ǩ�A�	* 

Average reward per step����Q       ��2	8���ǩ�A�	*

epsilon���N�z�.       ��W�	����ǩ�A�	* 

Average reward per step���F~�       ��2	����ǩ�A�	*

epsilon���yw��.       ��W�	đ�ǩ�A�	* 

Average reward per step���d��4       ��2	�đ�ǩ�A�	*

epsilon���0�I.       ��W�	5��ǩ�A�	* 

Average reward per step���ε�M       ��2	���ǩ�A�	*

epsilon���f��u.       ��W�	�N��ǩ�A�	* 

Average reward per step���oр       ��2	@O��ǩ�A�	*

epsilon���ƣ��.       ��W�	�̗�ǩ�A�	* 

Average reward per step����prB       ��2	�͗�ǩ�A�	*

epsilon����f1�.       ��W�	@���ǩ�A�	* 

Average reward per step���<���       ��2	����ǩ�A�	*

epsilon���/uS6.       ��W�	�,��ǩ�A�	* 

Average reward per step����a�+       ��2	�-��ǩ�A�	*

epsilon����ح$.       ��W�	�A��ǩ�A�
* 

Average reward per step�����R       ��2	�B��ǩ�A�
*

epsilon���P��.       ��W�	���ǩ�A�
* 

Average reward per step����U0       ��2	��ǩ�A�
*

epsilon���cmf�.       ��W�	�$��ǩ�A�
* 

Average reward per step����G�<       ��2	�%��ǩ�A�
*

epsilon���H�.       ��W�	KV��ǩ�A�
* 

Average reward per step���hs^�       ��2	�V��ǩ�A�
*

epsilon���m7).       ��W�	Aؤ�ǩ�A�
* 

Average reward per step�����[�       ��2	�ؤ�ǩ�A�
*

epsilon���eq�%.       ��W�	;:��ǩ�A�
* 

Average reward per step���l0Ħ       ��2	.;��ǩ�A�
*

epsilon���f��A.       ��W�	,a��ǩ�A�
* 

Average reward per step���E^��       ��2	�a��ǩ�A�
*

epsilon���!��5.       ��W�	���ǩ�A�
* 

Average reward per step���9O       ��2	����ǩ�A�
*

epsilon���Q��0       ���_	/��ǩ�A&*#
!
Average reward per episode  ��;q.       ��W�	���ǩ�A&*!

total reward per episode  %É HQ.       ��W�	2���ǩ�A�
* 

Average reward per step  ��mh       ��2	ު��ǩ�A�
*

epsilon  ���.       ��W�	���ǩ�A�
* 

Average reward per step  �Y�       ��2	���ǩ�A�
*

epsilon  ����.       ��W�	P��ǩ�A�
* 

Average reward per step  ��a�C       ��2	*��ǩ�A�
*

epsilon  �v��.       ��W�	%��ǩ�A�
* 

Average reward per step  �3ґ�       ��2	:��ǩ�A�
*

epsilon  ��p�.       ��W�	'���ǩ�A�
* 

Average reward per step  ���II       ��2	׉��ǩ�A�
*

epsilon  �kA�m.       ��W�	9���ǩ�A�
* 

Average reward per step  ��Q��       ��2	(���ǩ�A�
*

epsilon  �=��e.       ��W�	{���ǩ�A�
* 

Average reward per step  �J��       ��2	#���ǩ�A�
*

epsilon  ���B.       ��W�	tD��ǩ�A�
* 

Average reward per step  ���.       ��2	$E��ǩ�A�
*

epsilon  �3{~.       ��W�	����ǩ�A�
* 

Average reward per step  �� ��       ��2	����ǩ�A�
*

epsilon  ��h�.       ��W�	���ǩ�A�
* 

Average reward per step  �Z�S�       ��2	M��ǩ�A�
*

epsilon  �j`.       ��W�	�B��ǩ�A�
* 

Average reward per step  ��T�       ��2	�C��ǩ�A�
*

epsilon  �Nw�.       ��W�	�u��ǩ�A�
* 

Average reward per step  �Dd��       ��2	�v��ǩ�A�
*

epsilon  �{�<u.       ��W�	(��ǩ�A�
* 

Average reward per step  �Ż��       ��2	���ǩ�A�
*

epsilon  ���ZC.       ��W�	�l��ǩ�A�
* 

Average reward per step  ���'.       ��2	�m��ǩ�A�
*

epsilon  �V�X�.       ��W�	`��ǩ�A�
* 

Average reward per step  �R?�-       ��2	F��ǩ�A�
*

epsilon  ���L.       ��W�	�6��ǩ�A�
* 

Average reward per step  �3�       ��2	e7��ǩ�A�
*

epsilon  �~��.       ��W�	ρ��ǩ�A�
* 

Average reward per step  �j�       ��2	Â��ǩ�A�
*

epsilon  ���.       ��W�	:t��ǩ�A�
* 

Average reward per step  ���O�       ��2	!u��ǩ�A�
*

epsilon  �)6�".       ��W�	����ǩ�A�
* 

Average reward per step  �f���       ��2	����ǩ�A�
*

epsilon  �*�?J.       ��W�	�t��ǩ�A�
* 

Average reward per step  � #Ɉ       ��2	du��ǩ�A�
*

epsilon  ��n��.       ��W�	���ǩ�A�
* 

Average reward per step  ���       ��2	$���ǩ�A�
*

epsilon  ��4I.       ��W�	3��ǩ�A�
* 

Average reward per step  ��?��       ��2	��ǩ�A�
*

epsilon  �ڹW�.       ��W�	���ǩ�A�
* 

Average reward per step  �e�;       ��2	����ǩ�A�
*

epsilon  ��{��.       ��W�	b���ǩ�A�
* 

Average reward per step  �L�	�       ��2	b���ǩ�A�
*

epsilon  ���C�.       ��W�	ؚ��ǩ�A�
* 

Average reward per step  �G5�       ��2	����ǩ�A�
*

epsilon  ��D5�.       ��W�	7���ǩ�A�
* 

Average reward per step  ��\�g       ��2	���ǩ�A�
*

epsilon  ��{��.       ��W�	����ǩ�A�
* 

Average reward per step  �D       ��2	����ǩ�A�
*

epsilon  ��8V�.       ��W�	�<��ǩ�A�
* 

Average reward per step  ��       ��2	�=��ǩ�A�
*

epsilon  ��;��.       ��W�	]���ǩ�A�
* 

Average reward per step  ��Q�=       ��2	L���ǩ�A�
*

epsilon  �LN�_.       ��W�	e���ǩ�A�
* 

Average reward per step  �2�`�       ��2	L���ǩ�A�
*

epsilon  �sP��.       ��W�	����ǩ�A�
* 

Average reward per step  �ߚ��       ��2	F���ǩ�A�
*

epsilon  �]C\=.       ��W�	���ǩ�A�
* 

Average reward per step  �����       ��2	;��ǩ�A�
*

epsilon  ���8.       ��W�	����ǩ�A�
* 

Average reward per step  ��s�W       ��2	l���ǩ�A�
*

epsilon  �cgX,0       ���_	���ǩ�A'*#
!
Average reward per episode����#N/F.       ��W�	����ǩ�A'*!

total reward per episode  ��{^.       ��W�	�a��ǩ�A�
* 

Average reward per step����^yLy       ��2	�b��ǩ�A�
*

epsilon����{���.       ��W�	A���ǩ�A�
* 

Average reward per step����q���       ��2	���ǩ�A�
*

epsilon����.��$.       ��W�	�1��ǩ�A�
* 

Average reward per step�����O�       ��2	�2��ǩ�A�
*

epsilon�����s"�.       ��W�	UL �ǩ�A�
* 

Average reward per step������~�       ��2	M �ǩ�A�
*

epsilon����X��.       ��W�	���ǩ�A�
* 

Average reward per step������z�       ��2	���ǩ�A�
*

epsilon�����y��.       ��W�	� �ǩ�A�
* 

Average reward per step�����E        ��2	�!�ǩ�A�
*

epsilon����D�:�.       ��W�	�M�ǩ�A�
* 

Average reward per step����4��z       ��2	�N�ǩ�A�
*

epsilon�����_ .       ��W�	~t�ǩ�A�
* 

Average reward per step����fu       ��2	Su�ǩ�A�
*

epsilon�����K�.       ��W�	��
�ǩ�A�
* 

Average reward per step����E5x       ��2	*�
�ǩ�A�
*

epsilon����H��.       ��W�	���ǩ�A�
* 

Average reward per step�������       ��2	o��ǩ�A�
*

epsilon�����3).       ��W�	��ǩ�A�
* 

Average reward per step�����%�       ��2	*�ǩ�A�
*

epsilon�����f�.       ��W�	֭�ǩ�A�
* 

Average reward per step����x+��       ��2	���ǩ�A�
*

epsilon�����V�7.       ��W�	?6�ǩ�A�
* 

Average reward per step�������       ��2	�6�ǩ�A�
*

epsilon�����Uf�.       ��W�	�G�ǩ�A�
* 

Average reward per step��������       ��2	ZH�ǩ�A�
*

epsilon������0[.       ��W�	�Z�ǩ�A�
* 

Average reward per step����~ҳ       ��2	�[�ǩ�A�
*

epsilon������gV.       ��W�	���ǩ�A�
* 

Average reward per step����ъ`]       ��2	��ǩ�A�
*

epsilon����E��G.       ��W�	���ǩ�A�
* 

Average reward per step��������       ��2	���ǩ�A�
*

epsilon����,L.       ��W�	���ǩ�A�
* 

Average reward per step����{���       ��2	{��ǩ�A�
*

epsilon�����iQ.       ��W�	'��ǩ�A�
* 

Average reward per step�������       ��2	��ǩ�A�
*

epsilon����!Y�.       ��W�	o!�ǩ�A�
* 

Average reward per step����1�SR       ��2	E!�ǩ�A�
*

epsilon����B$�.       ��W�	Z�"�ǩ�A�
* 

Average reward per step�����С+       ��2	4�"�ǩ�A�
*

epsilon����K�ކ.       ��W�	� %�ǩ�A�
* 

Average reward per step����8�j       ��2	2!%�ǩ�A�
*

epsilon�������.       ��W�	Ĳ&�ǩ�A�
* 

Average reward per step������м       ��2	p�&�ǩ�A�
*

epsilon�����9u�.       ��W�	��(�ǩ�A�
* 

Average reward per step����7��B       ��2	��(�ǩ�A�
*

epsilon������g�0       ���_	W")�ǩ�A(*#
!
Average reward per episode  �����.       ��W�	h#)�ǩ�A(*!

total reward per episode  %�?7��.       ��W�	f/�ǩ�A�
* 

Average reward per step  ��
x��       ��2	/�ǩ�A�
*

epsilon  ��7}�.       ��W�	�X1�ǩ�A�
* 

Average reward per step  ���H��       ��2	�Y1�ǩ�A�
*

epsilon  �����.       ��W�	�{3�ǩ�A�
* 

Average reward per step  ����Q       ��2	�|3�ǩ�A�
*

epsilon  ������.       ��W�	�45�ǩ�A�
* 

Average reward per step  ��}g�       ��2	i55�ǩ�A�
*

epsilon  ����'.       ��W�	�v7�ǩ�A�
* 

Average reward per step  ���Lp4       ��2	}w7�ǩ�A�
*

epsilon  ���x��.       ��W�	z�9�ǩ�A�
* 

Average reward per step  ��\� �       ��2	a�9�ǩ�A�
*

epsilon  ����Q�.       ��W�	��;�ǩ�A�
* 

Average reward per step  ����J1       ��2	��;�ǩ�A�
*

epsilon  ��%詢.       ��W�	=�ǩ�A�
* 

Average reward per step  ��$_�       ��2	�=�ǩ�A�
*

epsilon  ���Z}�.       ��W�	�I?�ǩ�A�
* 

Average reward per step  ����>       ��2	�J?�ǩ�A�
*

epsilon  ���r*.       ��W�	 _A�ǩ�A�
* 

Average reward per step  ���fr�       ��2	�_A�ǩ�A�
*

epsilon  ��yn��.       ��W�	cC�ǩ�A�
* 

Average reward per step  ���hy       ��2	�cC�ǩ�A�
*

epsilon  ����.       ��W�	��E�ǩ�A�
* 

Average reward per step  ���>�       ��2	{�E�ǩ�A�
*

epsilon  ��L8�Y.       ��W�	�G�ǩ�A�
* 

Average reward per step  �����0       ��2	ڮG�ǩ�A�
*

epsilon  ��yN�.       ��W�	"�I�ǩ�A�
* 

Average reward per step  ����        ��2	��I�ǩ�A�
*

epsilon  ���ƨ+.       ��W�	^�K�ǩ�A�
* 

Average reward per step  ��f�l�       ��2	��K�ǩ�A�
*

epsilon  ��N�{�.       ��W�	�N�ǩ�A�
* 

Average reward per step  ��}UC       ��2	WN�ǩ�A�
*

epsilon  ��k��.       ��W�	UgO�ǩ�A�
* 

Average reward per step  ����       ��2	hO�ǩ�A�
*

epsilon  ���.��.       ��W�	ˢQ�ǩ�A�
* 

Average reward per step  ��E]_V       ��2	b�Q�ǩ�A�
*

epsilon  ���|�i.       ��W�	��S�ǩ�A�
* 

Average reward per step  ��j�a       ��2	~�S�ǩ�A�
*

epsilon  ���+�.       ��W�	w�U�ǩ�A�
* 

Average reward per step  ����Z�       ��2	I�U�ǩ�A�
*

epsilon  ��]M50       ���_	�U�ǩ�A)*#
!
Average reward per episode  �s�.       ��W�	��U�ǩ�A)*!

total reward per episode  %Á���.       ��W�	�'Z�ǩ�A�
* 

Average reward per step  �W���       ��2	V(Z�ǩ�A�
*

epsilon  �-p�5.       ��W�	�@\�ǩ�A�
* 

Average reward per step  �a�       ��2	pA\�ǩ�A�
*

epsilon  ����.       ��W�	)�]�ǩ�A�
* 

Average reward per step  �?�I       ��2	��]�ǩ�A�
*

epsilon  �t.       ��W�	�`�ǩ�A�
* 

Average reward per step  ����(       ��2	�`�ǩ�A�
*

epsilon  �`���.       ��W�	�Sb�ǩ�A�
* 

Average reward per step  ���'       ��2	eTb�ǩ�A�
*

epsilon  ���.       ��W�	��d�ǩ�A�
* 

Average reward per step  �]��1       ��2	.�d�ǩ�A�
*

epsilon  �3>s.       ��W�	֩f�ǩ�A�
* 

Average reward per step  �^�4C       ��2	m�f�ǩ�A�
*

epsilon  �F��[.       ��W�	��h�ǩ�A�
* 

Average reward per step  �̪       ��2	*�h�ǩ�A�
*

epsilon  �gR7>.       ��W�	��j�ǩ�A�
* 

Average reward per step  �^�t�       ��2	t�j�ǩ�A�
*

epsilon  �x�E�.       ��W�	�Tl�ǩ�A�
* 

Average reward per step  ���B       ��2	CUl�ǩ�A�
*

epsilon  ��6�j.       ��W�	Pqn�ǩ�A�
* 

Average reward per step  �O؆�       ��2	2rn�ǩ�A�
*

epsilon  �kzk�.       ��W�	�p�ǩ�A�
* 

Average reward per step  �?)��       ��2	��p�ǩ�A�
*

epsilon  ��6:K.       ��W�	�r�ǩ�A�
* 

Average reward per step  ��.�H       ��2	��r�ǩ�A�
*

epsilon  ��R�.       ��W�	��t�ǩ�A�
* 

Average reward per step  ���       ��2	��t�ǩ�A�
*

epsilon  ��Ub.       ��W�	>�v�ǩ�A�
* 

Average reward per step  ��Ŵm       ��2	!�v�ǩ�A�
*

epsilon  �Af3.       ��W�	�y�ǩ�A�
* 

Average reward per step  �2��m       ��2	�y�ǩ�A�
*

epsilon  ����.       ��W�	L{�ǩ�A�
* 

Average reward per step  ��}�y       ��2	z{�ǩ�A�
*

epsilon  �w�6�.       ��W�	8-}�ǩ�A�
* 

Average reward per step  ��	1�       ��2	�-}�ǩ�A�
*

epsilon  ���'.       ��W�	�G�ǩ�A�
* 

Average reward per step  ����       ��2	�H�ǩ�A�
*

epsilon  ���)(.       ��W�	 o��ǩ�A�
* 

Average reward per step  ���M�       ��2	"p��ǩ�A�
*

epsilon  �H�.       ��W�	�т�ǩ�A�
* 

Average reward per step  ���m       ��2	k҂�ǩ�A�
*

epsilon  ����.       ��W�	0L��ǩ�A�
* 

Average reward per step  �y~n       ��2	M��ǩ�A�
*

epsilon  ����.       ��W�	�i��ǩ�A�
* 

Average reward per step  ���o�       ��2	nj��ǩ�A�
*

epsilon  �g� .       ��W�	I���ǩ�A�
* 

Average reward per step  ��)�       ��2	���ǩ�A�
*

epsilon  �d�>.       ��W�	ș��ǩ�A�
* 

Average reward per step  ���Dm       ��2	g���ǩ�A�
*

epsilon  �[�C_.       ��W�	���ǩ�A�
* 

Average reward per step  ��iU�       ��2	���ǩ�A�
*

epsilon  ����.       ��W�	��ǩ�A�
* 

Average reward per step  ��]�       ��2	*��ǩ�A�
*

epsilon  ����).       ��W�	y$��ǩ�A�
* 

Average reward per step  ���pE       ��2	%��ǩ�A�
*

epsilon  �E<��.       ��W�	tԓ�ǩ�A�
* 

Average reward per step  ��ĬG       ��2	Փ�ǩ�A�
*

epsilon  �Q�B.       ��W�	���ǩ�A�
* 

Average reward per step  �]dGW       ��2	���ǩ�A�
*

epsilon  �Cmg�.       ��W�	�t��ǩ�A�
* 

Average reward per step  ��.�       ��2	�u��ǩ�A�
*

epsilon  �ՏB.       ��W�	����ǩ�A�
* 

Average reward per step  ��ǋ       ��2	��ǩ�A�
*

epsilon  �_�k�.       ��W�	K���ǩ�A�
* 

Average reward per step  �U�.�       ��2	���ǩ�A�
*

epsilon  ��W[�.       ��W�	Ҭ��ǩ�A�
* 

Average reward per step  �7�~�       ��2	q���ǩ�A�
*

epsilon  ���ڃ.       ��W�	uʟ�ǩ�A�
* 

Average reward per step  �xd�6       ��2	˟�ǩ�A�
*

epsilon  �����.       ��W�	�ܡ�ǩ�A�
* 

Average reward per step  �uc       ��2	�ݡ�ǩ�A�
*

epsilon  �N��3.       ��W�	���ǩ�A�
* 

Average reward per step  ��qC�       ��2	:��ǩ�A�
*

epsilon  ���^.       ��W�	���ǩ�A�
* 

Average reward per step  �����       ��2	_��ǩ�A�
*

epsilon  � |�.       ��W�	����ǩ�A�
* 

Average reward per step  ��Ag�       ��2	����ǩ�A�
*

epsilon  ��y��.       ��W�	R��ǩ�A�
* 

Average reward per step  �Tb��       ��2	���ǩ�A�
*

epsilon  ��^u.       ��W�	�,��ǩ�A�
* 

Average reward per step  �TY{n       ��2	�-��ǩ�A�
*

epsilon  �(�|.       ��W�	_`��ǩ�A�
* 

Average reward per step  � �J       ��2	a��ǩ�A�
*

epsilon  ��F�.       ��W�	c|��ǩ�A�
* 

Average reward per step  �c���       ��2	=}��ǩ�A�
*

epsilon  �fD�.       ��W�	����ǩ�A�* 

Average reward per step  �17z8       ��2	N���ǩ�A�*

epsilon  ����.       ��W�	�	��ǩ�A�* 

Average reward per step  �7S��       ��2	�
��ǩ�A�*

epsilon  ��3��.       ��W�	C��ǩ�A�* 

Average reward per step  �C���       ��2	! ��ǩ�A�*

epsilon  �.ȳE.       ��W�	�J��ǩ�A�* 

Average reward per step  ��|O       ��2	�K��ǩ�A�*

epsilon  �ׂK.       ��W�	�p��ǩ�A�* 

Average reward per step  ���       ��2	�q��ǩ�A�*

epsilon  �᥋!.       ��W�	�u��ǩ�A�* 

Average reward per step  ��&Zw       ��2	6v��ǩ�A�*

epsilon  ���.       ��W�	$���ǩ�A�* 

Average reward per step  �E�q       ��2	����ǩ�A�*

epsilon  �V>��.       ��W�	����ǩ�A�* 

Average reward per step  ��L�5       ��2	_���ǩ�A�*

epsilon  ���X�.       ��W�	����ǩ�A�* 

Average reward per step  �%�:;       ��2	����ǩ�A�*

epsilon  �)�FY.       ��W�	#��ǩ�A�* 

Average reward per step  �
��.       ��2	�#��ǩ�A�*

epsilon  �z��).       ��W�	����ǩ�A�* 

Average reward per step  ����       ��2	t���ǩ�A�*

epsilon  ��16.       ��W�	;���ǩ�A�* 

Average reward per step  ��<�b       ��2	����ǩ�A�*

epsilon  �+��K.       ��W�	����ǩ�A�* 

Average reward per step  �ȉ[�       ��2	N���ǩ�A�*

epsilon  �א��.       ��W�	�	��ǩ�A�* 

Average reward per step  ����r       ��2	l
��ǩ�A�*

epsilon  ��ѥ�.       ��W�	&��ǩ�A�* 

Average reward per step  �BGޣ       ��2	�&��ǩ�A�*

epsilon  ���N�.       ��W�	����ǩ�A�* 

Average reward per step  ��V��       ��2	;���ǩ�A�*

epsilon  �^�`.       ��W�	{���ǩ�A�* 

Average reward per step  �� 1       ��2	���ǩ�A�*

epsilon  ����g.       ��W�	���ǩ�A�* 

Average reward per step  ���D�       ��2	.��ǩ�A�*

epsilon  ��3��.       ��W�	�A��ǩ�A�* 

Average reward per step  �iE��       ��2	�B��ǩ�A�*

epsilon  �k���.       ��W�	Mf��ǩ�A�* 

Average reward per step  ��P��       ��2	�f��ǩ�A�*

epsilon  ��
��.       ��W�	Kv��ǩ�A�* 

Average reward per step  �/��       ��2	�v��ǩ�A�*

epsilon  ����.       ��W�	����ǩ�A�* 

Average reward per step  ���/       ��2	����ǩ�A�*

epsilon  � �C.       ��W�	i6��ǩ�A�* 

Average reward per step  ����R       ��2	�6��ǩ�A�*

epsilon  ��Y�.       ��W�	`��ǩ�A�* 

Average reward per step  ���3�       ��2	�`��ǩ�A�*

epsilon  �B4�".       ��W�	؁��ǩ�A�* 

Average reward per step  ����b       ��2	����ǩ�A�*

epsilon  ���Bw.       ��W�	���ǩ�A�* 

Average reward per step  �0�f�       ��2	����ǩ�A�*

epsilon  �v�vA.       ��W�	`���ǩ�A�* 

Average reward per step  ���u       ��2	-���ǩ�A�*

epsilon  ����D.       ��W�	���ǩ�A�* 

Average reward per step  ��       ��2	���ǩ�A�*

epsilon  �)��.       ��W�	h���ǩ�A�* 

Average reward per step  ��Iei       ��2	G���ǩ�A�*

epsilon  ����.       ��W�	���ǩ�A�* 

Average reward per step  �&/0�       ��2	¥��ǩ�A�*

epsilon  �!D�k.       ��W�	P���ǩ�A�* 

Average reward per step  ��O=       ��2	����ǩ�A�*

epsilon  �NRp.       ��W�	����ǩ�A�* 

Average reward per step  ��8X�       ��2	����ǩ�A�*

epsilon  ���R.       ��W�	Af��ǩ�A�* 

Average reward per step  �W�S       ��2	g��ǩ�A�*

epsilon  ���X�.       ��W�	"���ǩ�A�* 

Average reward per step  �ʪs�       ��2	����ǩ�A�*

epsilon  ���#.       ��W�	i���ǩ�A�* 

Average reward per step  �s��c       ��2	7���ǩ�A�*

epsilon  �+��S.       ��W�	����ǩ�A�* 

Average reward per step  ��&[W       ��2	:���ǩ�A�*

epsilon  ��jP�.       ��W�	a���ǩ�A�* 

Average reward per step  �\:�       ��2	���ǩ�A�*

epsilon  ����?.       ��W�	���ǩ�A�* 

Average reward per step  ����       ��2	: ��ǩ�A�*

epsilon  ��i�u.       ��W�	0K��ǩ�A�* 

Average reward per step  �˓�       ��2	�K��ǩ�A�*

epsilon  ��>.       ��W�	�L�ǩ�A�* 

Average reward per step  �kfy	       ��2	'M�ǩ�A�*

epsilon  �ekDf.       ��W�	��ǩ�A�* 

Average reward per step  ��<I       ��2	���ǩ�A�*

epsilon  �py2L.       ��W�	L6�ǩ�A�* 

Average reward per step  �r��       ��2	"7�ǩ�A�*

epsilon  ��@(.       ��W�	Ҍ�ǩ�A�* 

Average reward per step  ��uO9       ��2	���ǩ�A�*

epsilon  �����.       ��W�	M��ǩ�A�* 

Average reward per step  �/�#�       ��2	���ǩ�A�*

epsilon  �eN��.       ��W�	p>
�ǩ�A�* 

Average reward per step  ��Ӧ       ��2	>?
�ǩ�A�*

epsilon  �)���.       ��W�	���ǩ�A�* 

Average reward per step  ��M        ��2	ڌ�ǩ�A�*

epsilon  ���Q�0       ���_	G��ǩ�A**#
!
Average reward per episode�f>.��.       ��W�	���ǩ�A**!

total reward per episode  �A!3�f.       ��W�	�O�ǩ�A�* 

Average reward per step�f>�a߼       ��2	�P�ǩ�A�*

epsilon�f>��`/.       ��W�	:x�ǩ�A�* 

Average reward per step�f>���N       ��2	y�ǩ�A�*

epsilon�f>,4`.       ��W�	{3�ǩ�A�* 

Average reward per step�f>x���       ��2	n4�ǩ�A�*

epsilon�f>^Fi�.       ��W�	ȴ�ǩ�A�* 

Average reward per step�f>#�[       ��2	p��ǩ�A�*

epsilon�f>i0��.       ��W�	��ǩ�A�* 

Average reward per step�f>\�%       ��2	���ǩ�A�*

epsilon�f>8.�c.       ��W�	��ǩ�A�* 

Average reward per step�f>���J       ��2	r�ǩ�A�*

epsilon�f>�]x".       ��W�	 ��ǩ�A�* 

Average reward per step�f>b��{       ��2	���ǩ�A�*

epsilon�f>0���.       ��W�	]S�ǩ�A�* 

Average reward per step�f>;�C�       ��2	T�ǩ�A�*

epsilon�f>Ӽ�.       ��W�	x� �ǩ�A�* 

Average reward per step�f>*�^K       ��2	� �ǩ�A�*

epsilon�f>++.       ��W�	E�"�ǩ�A�* 

Average reward per step�f>��:/       ��2	�"�ǩ�A�*

epsilon�f>cݟ�.       ��W�	� %�ǩ�A�* 

Average reward per step�f>{Dv       ��2	�!%�ǩ�A�*

epsilon�f>(��d.       ��W�	�x'�ǩ�A�* 

Average reward per step�f>e���       ��2	�y'�ǩ�A�*

epsilon�f>��	�.       ��W�	q+�ǩ�A�* 

Average reward per step�f>��.�       ��2	6 +�ǩ�A�*

epsilon�f>)�.       ��W�	fL-�ǩ�A�* 

Average reward per step�f>f�~       ��2	M-�ǩ�A�*

epsilon�f>.�.       ��W�	yv/�ǩ�A�* 

Average reward per step�f>,�9       ��2	Ow/�ǩ�A�*

epsilon�f>���0       ���_	V�/�ǩ�A+*#
!
Average reward per episodeUU5�6`�.       ��W�	�/�ǩ�A+*!

total reward per episode  *�n�:0.       ��W�	Yl3�ǩ�A�* 

Average reward per stepUU5����       ��2	;m3�ǩ�A�*

epsilonUU5�ˆB�.       ��W�	,�5�ǩ�A�* 

Average reward per stepUU5��;       ��2	��5�ǩ�A�*

epsilonUU5����<.       ��W�	�*7�ǩ�A�* 

Average reward per stepUU5���b       ��2	�+7�ǩ�A�*

epsilonUU5�t��.       ��W�	aT9�ǩ�A�* 

Average reward per stepUU5��`�       ��2	2U9�ǩ�A�*

epsilonUU5��̫.       ��W�	��;�ǩ�A�* 

Average reward per stepUU5�|/�Z       ��2	��;�ǩ�A�*

epsilonUU5��)h.       ��W�	��=�ǩ�A�* 

Average reward per stepUU5�H��       ��2	|�=�ǩ�A�*

epsilonUU5��;U.       ��W�	fM?�ǩ�A�* 

Average reward per stepUU5�J9�r       ��2	N?�ǩ�A�*

epsilonUU5�f�.       ��W�	�iA�ǩ�A�* 

Average reward per stepUU5�(���       ��2	�jA�ǩ�A�*

epsilonUU5�B��.       ��W�	j�C�ǩ�A�* 

Average reward per stepUU5����       ��2	�C�ǩ�A�*

epsilonUU5�]�h�.       ��W�	��E�ǩ�A�* 

Average reward per stepUU5�'(��       ��2	c�E�ǩ�A�*

epsilonUU5���Ų.       ��W�	ʌH�ǩ�A�* 

Average reward per stepUU5���       ��2	ʍH�ǩ�A�*

epsilonUU5����.       ��W�	�L�ǩ�A�* 

Average reward per stepUU5��$Mv       ��2	�L�ǩ�A�*

epsilonUU5�ez�.       ��W�	]6N�ǩ�A�* 

Average reward per stepUU5�L��9       ��2	C7N�ǩ�A�*

epsilonUU5���K.       ��W�	CQ�ǩ�A�* 

Average reward per stepUU5���g       ��2	iQ�ǩ�A�*

epsilonUU5����.       ��W�	R�T�ǩ�A�* 

Average reward per stepUU5�F�h       ��2	A�T�ǩ�A�*

epsilonUU5�g/�d.       ��W�	ƉV�ǩ�A�* 

Average reward per stepUU5����o       ��2	��V�ǩ�A�*

epsilonUU5���.       ��W�	"�Z�ǩ�A�* 

Average reward per stepUU5���Y�       ��2	�Z�ǩ�A�*

epsilonUU5��tU|.       ��W�	�]�ǩ�A�* 

Average reward per stepUU5�^�Vu       ��2	�]�ǩ�A�*

epsilonUU5��8�q.       ��W�	:�^�ǩ�A�* 

Average reward per stepUU5�Ry��       ��2	��^�ǩ�A�*

epsilonUU5�pW��.       ��W�	�&a�ǩ�A�* 

Average reward per stepUU5�eE��       ��2	�'a�ǩ�A�*

epsilonUU5���.{.       ��W�	�b�ǩ�A�* 

Average reward per stepUU5���U�       ��2	��b�ǩ�A�*

epsilonUU5��;t�.       ��W�	8�d�ǩ�A�* 

Average reward per stepUU5��Ss�       ��2	�d�ǩ�A�*

epsilonUU5�d��N.       ��W�	�Lg�ǩ�A�* 

Average reward per stepUU5�����       ��2	rMg�ǩ�A�*

epsilonUU5�x�jh.       ��W�	k�ǩ�A�* 

Average reward per stepUU5��ڼ�       ��2	�k�ǩ�A�*

epsilonUU5�5T��0       ���_	 8k�ǩ�A,*#
!
Average reward per episodeUU����T�.       ��W�	�8k�ǩ�A,*!

total reward per episode  ��/�.       ��W�	�Mo�ǩ�A�* 

Average reward per stepUU���S�       ��2	�No�ǩ�A�*

epsilonUU��Q}�.       ��W�	[
q�ǩ�A�* 

Average reward per stepUU���a��       ��2	1q�ǩ�A�*

epsilonUU���F!M.       ��W�	�[s�ǩ�A�* 

Average reward per stepUU����J�       ��2	�\s�ǩ�A�*

epsilonUU��n��.       ��W�	)�u�ǩ�A�* 

Average reward per stepUU�����k       ��2	��u�ǩ�A�*

epsilonUU��v( .       ��W�	�Qw�ǩ�A�* 

Average reward per stepUU��Z�y       ��2	�Rw�ǩ�A�*

epsilonUU���ĩ..       ��W�	g�y�ǩ�A�* 

Average reward per stepUU����w�       ��2	��y�ǩ�A�*

epsilonUU���
Y/.       ��W�	ė{�ǩ�A�* 

Average reward per stepUU����U�       ��2	��{�ǩ�A�*

epsilonUU��:%
.       ��W�	�~�ǩ�A�* 

Average reward per stepUU���7S�       ��2	~�ǩ�A�*

epsilonUU�����C.       ��W�	Y��ǩ�A�* 

Average reward per stepUU��a�T�       ��2	3��ǩ�A�*

epsilonUU���u�.       ��W�	����ǩ�A�* 

Average reward per stepUU���\W�       ��2	x���ǩ�A�*

epsilonUU������.       ��W�	7 ��ǩ�A�* 

Average reward per stepUU���>��       ��2	��ǩ�A�*

epsilonUU��p}��.       ��W�	Ed��ǩ�A�* 

Average reward per stepUU���CX       ��2	(e��ǩ�A�*

epsilonUU����.       ��W�	(���ǩ�A�* 

Average reward per stepUU��ً.+       ��2	���ǩ�A�*

epsilonUU��i�_U.       ��W�	"��ǩ�A�* 

Average reward per stepUU��IE�       ��2	��ǩ�A�*

epsilonUU��<�G�.       ��W�	�J��ǩ�A�* 

Average reward per stepUU���*�e       ��2	�K��ǩ�A�*

epsilonUU�����.       ��W�	6���ǩ�A�* 

Average reward per stepUU��;�       ��2	[���ǩ�A�*

epsilonUU�����.       ��W�	���ǩ�A�* 

Average reward per stepUU��N���       ��2	���ǩ�A�*

epsilonUU����j�.       ��W�	�1��ǩ�A�* 

Average reward per stepUU���.�8       ��2	/3��ǩ�A�*

epsilonUU��=+_�.       ��W�	����ǩ�A�* 

Average reward per stepUU���J       ��2	���ǩ�A�*

epsilonUU����..       ��W�	���ǩ�A�* 

Average reward per stepUU���bK       ��2	���ǩ�A�*

epsilonUU����"�.       ��W�	
٘�ǩ�A�* 

Average reward per stepUU��M��1       ��2	�٘�ǩ�A�*

epsilonUU������.       ��W�	]N��ǩ�A�* 

Average reward per stepUU���s�8       ��2	<O��ǩ�A�*

epsilonUU��}<�.       ��W�	Ė��ǩ�A�* 

Average reward per stepUU��K�Sw       ��2	����ǩ�A�*

epsilonUU�����v.       ��W�	J��ǩ�A�* 

Average reward per stepUU��p�X	       ��2	(��ǩ�A�*

epsilonUU����.       ��W�	�ݢ�ǩ�A�* 

Average reward per stepUU���:��       ��2	�ޢ�ǩ�A�*

epsilonUU��䇠�.       ��W�		6��ǩ�A�* 

Average reward per stepUU���H{=       ��2	�6��ǩ�A�*

epsilonUU���u�.       ��W�	o֧�ǩ�A�* 

Average reward per stepUU����0�       ��2	�ק�ǩ�A�*

epsilonUU��'9�$.       ��W�	a���ǩ�A�* 

Average reward per stepUU���≺       ��2	 ���ǩ�A�*

epsilonUU��UF[�.       ��W�	|��ǩ�A�* 

Average reward per stepUU��u�
F       ��2	t��ǩ�A�*

epsilonUU����l|.       ��W�	�ʯ�ǩ�A�* 

Average reward per stepUU���`�       ��2	�˯�ǩ�A�*

epsilonUU����t�.       ��W�	X��ǩ�A�* 

Average reward per stepUU���e       ��2	! ��ǩ�A�*

epsilonUU��D��.       ��W�	�H��ǩ�A�* 

Average reward per stepUU�����F       ��2	�I��ǩ�A�*

epsilonUU��գ�.       ��W�	>��ǩ�A�* 

Average reward per stepUU���mr       ��2	��ǩ�A�*

epsilonUU�� ��.       ��W�	g���ǩ�A�* 

Average reward per stepUU����;�       ��2	���ǩ�A�*

epsilonUU������.       ��W�	�Ƽ�ǩ�A�* 

Average reward per stepUU��5��       ��2	�Ǽ�ǩ�A�*

epsilonUU��b&.       ��W�	���ǩ�A�* 

Average reward per stepUU���/        ��2	`��ǩ�A�*

epsilonUU��.�\4.       ��W�	V��ǩ�A�* 

Average reward per stepUU�����       ��2	�V��ǩ�A�*

epsilonUU��r�.       ��W�	:>��ǩ�A�* 

Average reward per stepUU���X!P       ��2	?��ǩ�A�*

epsilonUU��J�,$.       ��W�	ʤ��ǩ�A�* 

Average reward per stepUU���g�       ��2	����ǩ�A�*

epsilonUU��{��.       ��W�	+���ǩ�A�* 

Average reward per stepUU���!;�       ��2	���ǩ�A�*

epsilonUU���|�.       ��W�	���ǩ�A�* 

Average reward per stepUU���awB       ��2	���ǩ�A�*

epsilonUU��x�Y.       ��W�	�N��ǩ�A�* 

Average reward per stepUU���Y�       ��2	]O��ǩ�A�*

epsilonUU��N���.       ��W�	����ǩ�A�* 

Average reward per stepUU��pŽ       ��2	n���ǩ�A�*

epsilonUU���N.       ��W�	�d��ǩ�A�* 

Average reward per stepUU����D�       ��2	�e��ǩ�A�*

epsilonUU���qS�.       ��W�	c'��ǩ�A�* 

Average reward per stepUU��]�#v       ��2	9(��ǩ�A�*

epsilonUU�����(.       ��W�	����ǩ�A�* 

Average reward per stepUU���Y��       ��2	����ǩ�A�*

epsilonUU���>�.       ��W�	l���ǩ�A�* 

Average reward per stepUU��A.�N       ��2	%���ǩ�A�*

epsilonUU��f��l.       ��W�	� ��ǩ�A�* 

Average reward per stepUU��U v�       ��2	���ǩ�A�*

epsilonUU���
�/.       ��W�	5��ǩ�A�* 

Average reward per stepUU����ȿ       ��2	�5��ǩ�A�*

epsilonUU����s`.       ��W�	���ǩ�A�* 

Average reward per stepUU��q���       ��2	ǃ��ǩ�A�*

epsilonUU��X�Hk.       ��W�	�Z��ǩ�A�* 

Average reward per stepUU���}N�       ��2	�[��ǩ�A�*

epsilonUU��ؖ}�.       ��W�	f���ǩ�A�* 

Average reward per stepUU��{?�       ��2	+���ǩ�A�*

epsilonUU��Bޞ.       ��W�	P��ǩ�A�* 

Average reward per stepUU��)kZ�       ��2	i��ǩ�A�*

epsilonUU���bU$.       ��W�	m���ǩ�A�* 

Average reward per stepUU��>~?       ��2	!���ǩ�A�*

epsilonUU��Vkz.       ��W�	�X��ǩ�A�* 

Average reward per stepUU����<       ��2	�Y��ǩ�A�*

epsilonUU��;��.       ��W�	����ǩ�A�* 

Average reward per stepUU���@�i       ��2	8���ǩ�A�*

epsilonUU����'..       ��W�	����ǩ�A�* 

Average reward per stepUU��O_˚       ��2	ǡ��ǩ�A�*

epsilonUU��_E�.       ��W�	����ǩ�A�* 

Average reward per stepUU��8/g�       ��2	l���ǩ�A�*

epsilonUU����}�.       ��W�	6��ǩ�A�* 

Average reward per stepUU���Z       ��2	��ǩ�A�*

epsilonUU���`r�.       ��W�	�q��ǩ�A�* 

Average reward per stepUU����+�       ��2	�r��ǩ�A�*

epsilonUU��ާ�.       ��W�	�t��ǩ�A�* 

Average reward per stepUU���?�&       ��2	:u��ǩ�A�*

epsilonUU����j1.       ��W�	����ǩ�A�* 

Average reward per stepUU��kkk       ��2	?���ǩ�A�*

epsilonUU����.       ��W�	���ǩ�A�* 

Average reward per stepUU��k@��       ��2	����ǩ�A�*

epsilonUU���X��.       ��W�	{��ǩ�A�* 

Average reward per stepUU���e       ��2	4��ǩ�A�*

epsilonUU�����N.       ��W�	���ǩ�A�* 

Average reward per stepUU�����       ��2	5��ǩ�A�*

epsilonUU���V3`.       ��W�	j�ǩ�A�* 

Average reward per stepUU��!�&X       ��2	U�ǩ�A�*

epsilonUU��ɴ�r.       ��W�	�=�ǩ�A�* 

Average reward per stepUU��fJ��       ��2	y>�ǩ�A�*

epsilonUU���,.       ��W�	�	�ǩ�A�* 

Average reward per stepUU��2�g       ��2	��	�ǩ�A�*

epsilonUU��x�FJ.       ��W�	��ǩ�A�* 

Average reward per stepUU�����|       ��2	H�ǩ�A�*

epsilonUU���DV.       ��W�	�-�ǩ�A�* 

Average reward per stepUU��r�(       ��2	�.�ǩ�A�*

epsilonUU��D���.       ��W�	2r�ǩ�A�* 

Average reward per stepUU������       ��2	s�ǩ�A�*

epsilonUU�����.       ��W�	<��ǩ�A�* 

Average reward per stepUU����/_       ��2	H��ǩ�A�*

epsilonUU�� �)�.       ��W�	7n�ǩ�A�* 

Average reward per stepUU��o�       ��2	�n�ǩ�A�*

epsilonUU���s�.       ��W�	���ǩ�A�* 

Average reward per stepUU��VcŹ       ��2	_��ǩ�A�*

epsilonUU��ސ�.       ��W�	���ǩ�A�* 

Average reward per stepUU���g�       ��2	���ǩ�A�*

epsilonUU��
��.       ��W�	��ǩ�A�* 

Average reward per stepUU��r�q~       ��2	��ǩ�A�*

epsilonUU��$K�O.       ��W�	Ք�ǩ�A�* 

Average reward per stepUU���ꍩ       ��2	���ǩ�A�*

epsilonUU��]��q.       ��W�	���ǩ�A�* 

Average reward per stepUU��*�E�       ��2		��ǩ�A�*

epsilonUU��Mv��.       ��W�	h�!�ǩ�A�* 

Average reward per stepUU���"E       ��2	d�!�ǩ�A�*

epsilonUU��ؕ��.       ��W�	�%$�ǩ�A�* 

Average reward per stepUU��^���       ��2	�&$�ǩ�A�*

epsilonUU���b9T.       ��W�	�6&�ǩ�A�* 

Average reward per stepUU���Sf       ��2	;7&�ǩ�A�*

epsilonUU����Q�.       ��W�	SA(�ǩ�A�* 

Average reward per stepUU��h8��       ��2	�A(�ǩ�A�*

epsilonUU���ʊ�.       ��W�	��)�ǩ�A�* 

Average reward per stepUU��9�]�       ��2	U�)�ǩ�A�*

epsilonUU������.       ��W�	�,�ǩ�A�* 

Average reward per stepUU��G���       ��2	],�ǩ�A�*

epsilonUU��%/��.       ��W�	JE.�ǩ�A�* 

Average reward per stepUU��5+e       ��2	�E.�ǩ�A�*

epsilonUU���l��.       ��W�	[`0�ǩ�A�* 

Average reward per stepUU��A�$       ��2	=a0�ǩ�A�*

epsilonUU��`)<.       ��W�	�2�ǩ�A�* 

Average reward per stepUU�����       ��2	˝2�ǩ�A�*

epsilonUU��xdw.       ��W�	�4�ǩ�A�* 

Average reward per stepUU��O��       ��2	��4�ǩ�A�*

epsilonUU����v�.       ��W�	y�6�ǩ�A�* 

Average reward per stepUU��a�P       ��2	W�6�ǩ�A�*

epsilonUU���G?a.       ��W�	�J8�ǩ�A�* 

Average reward per stepUU��睈W       ��2	wK8�ǩ�A�*

epsilonUU����A.       ��W�	��:�ǩ�A�* 

Average reward per stepUU��U�[       ��2	A�:�ǩ�A�*

epsilonUU�����.       ��W�	r�<�ǩ�A�* 

Average reward per stepUU���       ��2	��<�ǩ�A�*

epsilonUU�����p.       ��W�	��>�ǩ�A�* 

Average reward per stepUU��}j�l       ��2	��>�ǩ�A�*

epsilonUU��ΐ�.       ��W�	��@�ǩ�A�* 

Average reward per stepUU���B�       ��2	�@�ǩ�A�*

epsilonUU��X� .       ��W�	\�B�ǩ�A�* 

Average reward per stepUU���ò�       ��2	��B�ǩ�A�*

epsilonUU��-W.       ��W�	$&E�ǩ�A�* 

Average reward per stepUU��<_��       ��2	 'E�ǩ�A�*

epsilonUU����D^.       ��W�	��F�ǩ�A�* 

Average reward per stepUU��l�B0       ��2	k�F�ǩ�A�*

epsilonUU������.       ��W�	T�H�ǩ�A�* 

Average reward per stepUU��|���       ��2	��H�ǩ�A�*

epsilonUU��ⴡ�.       ��W�	�L�ǩ�A�* 

Average reward per stepUU��C�Ƃ       ��2	�L�ǩ�A�*

epsilonUU���.kF.       ��W�	p&O�ǩ�A�* 

Average reward per stepUU������       ��2	J'O�ǩ�A�*

epsilonUU������.       ��W�	&�Q�ǩ�A�* 

Average reward per stepUU��)I       ��2	!�Q�ǩ�A�*

epsilonUU���{��.       ��W�	,+T�ǩ�A�* 

Average reward per stepUU������       ��2	,T�ǩ�A�*

epsilonUU����)�.       ��W�	��W�ǩ�A�* 

Average reward per stepUU���Sݣ       ��2	{�W�ǩ�A�*

epsilonUU��Ό��0       ���_	��W�ǩ�A-*#
!
Average reward per episode�n�0��.       ��W�	K X�ǩ�A-*!

total reward per episode  ��`��.       ��W�	[\�ǩ�A�* 

Average reward per step�n��p       ��2	,\�ǩ�A�*

epsilon�n���.       ��W�	��]�ǩ�A�* 

Average reward per step�n����!       ��2	��]�ǩ�A�*

epsilon�n�M��.       ��W�	�F`�ǩ�A�* 

Average reward per step�n����       ��2	�G`�ǩ�A�*

epsilon�n�8�>�.       ��W�	�a�ǩ�A�* 

Average reward per step�n��>�(       ��2	��a�ǩ�A�*

epsilon�n����.       ��W�	�Gd�ǩ�A�* 

Average reward per step�n���>d       ��2	�Hd�ǩ�A�*

epsilon�n�a6�n.       ��W�	xf�ǩ�A�* 

Average reward per step�n�:�       ��2	�f�ǩ�A�*

epsilon�n� wa.       ��W�	�:h�ǩ�A�* 

Average reward per step�n�R��|       ��2	T;h�ǩ�A�*

epsilon�n����.       ��W�	�`j�ǩ�A�* 

Average reward per step�n�ec�\       ��2	�aj�ǩ�A�*

epsilon�n��:�.       ��W�	j�k�ǩ�A�* 

Average reward per step�n����(       ��2	8�k�ǩ�A�*

epsilon�n����r.       ��W�	�n�ǩ�A�* 

Average reward per step�n���1�       ��2	�n�ǩ�A�*

epsilon�n�[�:�.       ��W�	Hp�ǩ�A�* 

Average reward per step�n���h
       ��2	AIp�ǩ�A�*

epsilon�n��Ek�.       ��W�	+r�ǩ�A�* 

Average reward per step�n�q�>�       ��2	/r�ǩ�A�*

epsilon�n�<��.       ��W�	�Bt�ǩ�A�* 

Average reward per step�n���       ��2	WCt�ǩ�A�*

epsilon�n����[.       ��W�	cav�ǩ�A�* 

Average reward per step�n���ż       ��2	�av�ǩ�A�*

epsilon�n��]	�0       ���_	�{v�ǩ�A.*#
!
Average reward per episode%I:�}��.       ��W�	R|v�ǩ�A.*!

total reward per episode  #ä�O�.       ��W�	�"z�ǩ�A�* 

Average reward per step%I:�U��#       ��2	�#z�ǩ�A�*

epsilon%I:��Hs.       ��W�	.9|�ǩ�A�* 

Average reward per step%I:����       ��2	:|�ǩ�A�*

epsilon%I:���R�.       ��W�	̸~�ǩ�A�* 

Average reward per step%I:��+�        ��2	��~�ǩ�A�*

epsilon%I:��j�.       ��W�	?���ǩ�A�* 

Average reward per step%I:�2k͒       ��2	���ǩ�A�*

epsilon%I:���8.       ��W�	g,��ǩ�A�* 

Average reward per step%I:�z�O       ��2	^-��ǩ�A�*

epsilon%I:�b�sp.       ��W�	�	��ǩ�A�* 

Average reward per step%I:��yRc       ��2	F
��ǩ�A�*

epsilon%I:��'��.       ��W�	����ǩ�A�* 

Average reward per step%I:���	
       ��2	j�ǩ�A�*

epsilon%I:����.       ��W�	�\��ǩ�A�* 

Average reward per step%I:�L��4       ��2	[]��ǩ�A�*

epsilon%I:�����.       ��W�	Z��ǩ�A�* 

Average reward per step%I:�zS�       ��2	(��ǩ�A�*

epsilon%I:���.       ��W�	�7��ǩ�A�* 

Average reward per step%I:�0�       ��2	�8��ǩ�A�*

epsilon%I:�}l�.       ��W�	vq��ǩ�A�* 

Average reward per step%I:���-�       ��2	r��ǩ�A�*

epsilon%I:�F���.       ��W�	%���ǩ�A�* 

Average reward per step%I:�Ba�       ��2	����ǩ�A�*

epsilon%I:�B�.       ��W�	S$��ǩ�A�* 

Average reward per step%I:�@��r       ��2	N%��ǩ�A�*

epsilon%I:�j��g.       ��W�	Fz��ǩ�A�* 

Average reward per step%I:���N�       ��2	J{��ǩ�A�*

epsilon%I:�ȹw�.       ��W�	�p��ǩ�A�* 

Average reward per step%I:��>�       ��2	�q��ǩ�A�*

epsilon%I:�LN�.       ��W�	^֛�ǩ�A�* 

Average reward per step%I:�q�N       ��2	,כ�ǩ�A�*

epsilon%I:�7��.       ��W�	���ǩ�A�* 

Average reward per step%I:�m�T       ��2	���ǩ�A�*

epsilon%I:��`z.       ��W�	��ǩ�A�* 

Average reward per step%I:�����       ��2	׊��ǩ�A�*

epsilon%I:��ˢ6.       ��W�	"ġ�ǩ�A�* 

Average reward per step%I:�v�!R       ��2	�ġ�ǩ�A�*

epsilon%I:��u�).       ��W�	�K��ǩ�A�* 

Average reward per step%I:�r�N       ��2	�L��ǩ�A�*

epsilon%I:���z.       ��W�	s���ǩ�A�* 

Average reward per step%I:��5�@       ��2	=���ǩ�A�*

epsilon%I:��]�.       ��W�	�զ�ǩ�A�* 

Average reward per step%I:�g�F�       ��2	�֦�ǩ�A�*

epsilon%I:�K0V(.       ��W�	����ǩ�A�* 

Average reward per step%I:��:       ��2	����ǩ�A�*

epsilon%I:�b) �.       ��W�	�ê�ǩ�A�* 

Average reward per step%I:�=��       ��2	�Ī�ǩ�A�*

epsilon%I:��Ս^.       ��W�	���ǩ�A�* 

Average reward per step%I:��b�$       ��2	���ǩ�A�*

epsilon%I:�'m�.       ��W�	x*��ǩ�A�* 

Average reward per step%I:�D�       ��2	+��ǩ�A�*

epsilon%I:�bN�.       ��W�	
���ǩ�A�* 

Average reward per step%I:�)�$�       ��2	໰�ǩ�A�*

epsilon%I:����.       ��W�	���ǩ�A�* 

Average reward per step%I:�O0T       ��2	a��ǩ�A�*

epsilon%I:�����.       ��W�	Y��ǩ�A�* 

Average reward per step%I:��t*t       ��2	��ǩ�A�*

epsilon%I:��P�9.       ��W�	G��ǩ�A�* 

Average reward per step%I:��g�       ��2	��ǩ�A�*

epsilon%I:���#O0       ���_	\8��ǩ�A/*#
!
Average reward per episode��������.       ��W�	�8��ǩ�A/*!

total reward per episode  ��g�.       ��W�	���ǩ�A�* 

Average reward per step������z       ��2	��ǩ�A�*

epsilon����+��.       ��W�	����ǩ�A�* 

Average reward per step����E�       ��2	`���ǩ�A�*

epsilon�����O.       ��W�	6>��ǩ�A�* 

Average reward per step����O��       ��2	)?��ǩ�A�*

epsilon����`9�.       ��W�	rk��ǩ�A�* 

Average reward per step����
�^�       ��2	@l��ǩ�A�*

epsilon������3.       ��W�	v���ǩ�A�* 

Average reward per step�������       ��2	���ǩ�A�*

epsilon�������,.       ��W�	%���ǩ�A�* 

Average reward per step����M�i�       ��2	���ǩ�A�*

epsilon����i��R.       ��W�	8/��ǩ�A�* 

Average reward per step����2��       ��2	0��ǩ�A�*

epsilon����y5[/.       ��W�	����ǩ�A�* 

Average reward per step����J�w�       ��2	Q���ǩ�A�*

epsilon����CI�.       ��W�	a���ǩ�A�* 

Average reward per step����C颻       ��2	m ��ǩ�A�*

epsilon������.       ��W�	T��ǩ�A�* 

Average reward per step����5s%       ��2	.��ǩ�A�*

epsilon�����TB�.       ��W�	x���ǩ�A�* 

Average reward per step�����
U       ��2	���ǩ�A�*

epsilon����(��E.       ��W�	���ǩ�A�* 

Average reward per step����}�N       ��2	���ǩ�A�*

epsilon�����'t.       ��W�	�>��ǩ�A�* 

Average reward per step�������       ��2	�?��ǩ�A�*

epsilon�����o.       ��W�	�s��ǩ�A�* 

Average reward per step�����y�       ��2	�t��ǩ�A�*

epsilon����QĂ�.       ��W�	=a��ǩ�A�* 

Average reward per step��������       ��2	�a��ǩ�A�*

epsilon����ʉA.       ��W�	i;��ǩ�A�* 

Average reward per step����=�Z       ��2	u<��ǩ�A�*

epsilon�����P��.       ��W�	����ǩ�A�* 

Average reward per step���� r��       ��2	����ǩ�A�*

epsilon�����PX.       ��W�	����ǩ�A�* 

Average reward per step�����Eo       ��2	|���ǩ�A�*

epsilon������N�.       ��W�	�~��ǩ�A�* 

Average reward per step�����So"       ��2	���ǩ�A�*

epsilon�����7��.       ��W�	���ǩ�A�* 

Average reward per step�����7Ew       ��2	����ǩ�A�*

epsilon�����pT�.       ��W�	=���ǩ�A�* 

Average reward per step����u�R&       ��2	9���ǩ�A�*

epsilon����Q|�.       ��W�	�!��ǩ�A�* 

Average reward per step�����d��       ��2	2"��ǩ�A�*

epsilon����B�p�.       ��W�	l���ǩ�A�* 

Average reward per step�����?       ��2	���ǩ�A�*

epsilon������0       ���_	%���ǩ�A0*#
!
Average reward per episode�����^��.       ��W�	����ǩ�A0*!

total reward per episode  ���X.       ��W�	�8��ǩ�A�* 

Average reward per step�������       ��2	C9��ǩ�A�*

epsilon������O'.       ��W�	|��ǩ�A�* 

Average reward per step����e�w�       ��2	}��ǩ�A�*

epsilon�����"ͤ.       ��W�	-!��ǩ�A�* 

Average reward per step����^�3�       ��2	:"��ǩ�A�*

epsilon����*c�v.       ��W�	����ǩ�A�* 

Average reward per step����pvw�       ��2	����ǩ�A�*

epsilon����4��}.       ��W�	���ǩ�A�* 

Average reward per step����2)c�       ��2	����ǩ�A�*

epsilon������7G.       ��W�	P��ǩ�A�* 

Average reward per step����d)X       ��2	�Q��ǩ�A�*

epsilon����ƽ��.       ��W�	y��ǩ�A�* 

Average reward per step��������       ��2	O��ǩ�A�*

epsilon����c)�C.       ��W�	W���ǩ�A�* 

Average reward per step����jd�       ��2	%���ǩ�A�*

epsilon����K��s.       ��W�	o��ǩ�A�* 

Average reward per step�������7       ��2	
��ǩ�A�*

epsilon�����d�.       ��W�	���ǩ�A�* 

Average reward per step�����7+�       ��2	���ǩ�A�*

epsilon�����E$.       ��W�	��ǩ�A�* 

Average reward per step������[�       ��2	���ǩ�A�*

epsilon�������3.       ��W�	��ǩ�A�* 

Average reward per step����J+!�       ��2	��ǩ�A�*

epsilon����Lڝ�.       ��W�	a
�ǩ�A�* 

Average reward per step�������       ��2	�a
�ǩ�A�*

epsilon����^<b�.       ��W�	�I�ǩ�A�* 

Average reward per step�������!       ��2	�J�ǩ�A�*

epsilon������Qe.       ��W�	aQ�ǩ�A�* 

Average reward per step����}��       ��2	R�ǩ�A�*

epsilon����o�Z.       ��W�	?��ǩ�A�* 

Average reward per step��������       ��2	���ǩ�A�*

epsilon������#�.       ��W�	B	�ǩ�A�* 

Average reward per step�����	       ��2	�	�ǩ�A�*

epsilon������Ί.       ��W�	|`�ǩ�A�* 

Average reward per step�������       ��2	�a�ǩ�A�*

epsilon�����w��.       ��W�	���ǩ�A�* 

Average reward per step����G)�-       ��2	s��ǩ�A�*

epsilon�����V�.       ��W�	u��ǩ�A�* 

Average reward per step����K���       ��2	:��ǩ�A�*

epsilon�������.       ��W�	;#�ǩ�A�* 

Average reward per step����D"�v       ��2	�#�ǩ�A�*

epsilon����n��.       ��W�	��$�ǩ�A�* 

Average reward per step������{�       ��2	��$�ǩ�A�*

epsilon�����..       ��W�	�0(�ǩ�A�* 

Average reward per step����'>_�       ��2	f1(�ǩ�A�*

epsilon����1A�m.       ��W�	;+�ǩ�A�* 

Average reward per step����:eAO       ��2	G<+�ǩ�A�*

epsilon����5��.       ��W�	g-�ǩ�A�* 

Average reward per step����JɊ/       ��2	�g-�ǩ�A�*

epsilon����^�*h.       ��W�	�	/�ǩ�A�* 

Average reward per step�����F�       ��2	�
/�ǩ�A�*

epsilon����7B.       ��W�	�J1�ǩ�A�* 

Average reward per step����E.V       ��2	0K1�ǩ�A�*

epsilon����A'�n.       ��W�	�p3�ǩ�A�* 

Average reward per step�������       ��2	Tq3�ǩ�A�*

epsilon������0       ���_	�3�ǩ�A1*#
!
Average reward per episode۶���#E�.       ��W�	��3�ǩ�A1*!

total reward per episode  �S�Y.       ��W�	��7�ǩ�A�* 

Average reward per step۶���B       ��2	Y�7�ǩ�A�*

epsilon۶��G���.       ��W�	"Q9�ǩ�A�* 

Average reward per step۶��z霄       ��2	�Q9�ǩ�A�*

epsilon۶����:2.       ��W�	;�ǩ�A�* 

Average reward per step۶����S�       ��2	;�ǩ�A�*

epsilon۶��S1.       ��W�	��<�ǩ�A�* 

Average reward per step۶������       ��2	8�<�ǩ�A�*

epsilon۶���P�.       ��W�	�>�ǩ�A�* 

Average reward per step۶�� c�	       ��2	n�>�ǩ�A�*

epsilon۶��c1a�.       ��W�	^HA�ǩ�A�* 

Average reward per step۶���5��       ��2	IIA�ǩ�A�*

epsilon۶���s�.       ��W�	��B�ǩ�A�* 

Average reward per step۶��\L�B       ��2	$�B�ǩ�A�*

epsilon۶��?���.       ��W�	�E�ǩ�A�* 

Average reward per step۶��<3c�       ��2	�E�ǩ�A�*

epsilon۶��& �.       ��W�	�1G�ǩ�A�* 

Average reward per step۶���X�       ��2	�2G�ǩ�A�*

epsilon۶���x��.       ��W�	|{I�ǩ�A�* 

Average reward per step۶��k��7       ��2	|I�ǩ�A�*

epsilon۶������.       ��W�	`K�ǩ�A�* 

Average reward per step۶�����       ��2	>K�ǩ�A�*

epsilon۶����).       ��W�	�"M�ǩ�A�* 

Average reward per step۶����#       ��2	�#M�ǩ�A�*

epsilon۶�����.       ��W�	�?O�ǩ�A�* 

Average reward per step۶��/I'�       ��2	�@O�ǩ�A�*

epsilon۶��BԦ�.       ��W�	bKQ�ǩ�A�* 

Average reward per step۶��ɇ��       ��2	8LQ�ǩ�A�*

epsilon۶���>P.       ��W�	N}S�ǩ�A�* 

Average reward per step۶��m`�       ��2	�}S�ǩ�A�*

epsilon۶��V%��.       ��W�	|�U�ǩ�A�* 

Average reward per step۶������       ��2	,�U�ǩ�A�*

epsilon۶���A�.       ��W�	m�W�ǩ�A�* 

Average reward per step۶����(w       ��2	X�W�ǩ�A�*

epsilon۶����T.       ��W�	��Y�ǩ�A�* 

Average reward per step۶��_ ��       ��2	��Y�ǩ�A�*

epsilon۶���]bf.       ��W�	��[�ǩ�A�* 

Average reward per step۶��6�2�       ��2	a�[�ǩ�A�*

epsilon۶����F�.       ��W�	�^�ǩ�A�* 

Average reward per step۶��
;�       ��2	�^�ǩ�A�*

epsilon۶��&T�.       ��W�	��_�ǩ�A�* 

Average reward per step۶���j0       ��2	��_�ǩ�A�*

epsilon۶��FI��.       ��W�	�b�ǩ�A�* 

Average reward per step۶�����       ��2	bb�ǩ�A�*

epsilon۶��F��.       ��W�	�Ad�ǩ�A�* 

Average reward per step۶��jBr       ��2	tBd�ǩ�A�*

epsilon۶����0'.       ��W�	��e�ǩ�A�* 

Average reward per step۶���|�,       ��2	��e�ǩ�A�*

epsilon۶���1.       ��W�	'Kh�ǩ�A�* 

Average reward per step۶���i�:       ��2	�Kh�ǩ�A�*

epsilon۶��d�i�.       ��W�	�qj�ǩ�A�* 

Average reward per step۶��χ�        ��2	~rj�ǩ�A�*

epsilon۶����o4.       ��W�	��k�ǩ�A�* 

Average reward per step۶���ו%       ��2	t�k�ǩ�A�*

epsilon۶��V�7.       ��W�	C n�ǩ�A�* 

Average reward per step۶��Y���       ��2	>!n�ǩ�A�*

epsilon۶���>�`.       ��W�	�Gp�ǩ�A�* 

Average reward per step۶������       ��2	�Hp�ǩ�A�*

epsilon۶��DB�F.       ��W�	dXr�ǩ�A�* 

Average reward per step۶��� �       ��2	KYr�ǩ�A�*

epsilon۶��L"�Z.       ��W�	��t�ǩ�A�* 

Average reward per step۶�����C       ��2	��t�ǩ�A�*

epsilon۶��,�pI.       ��W�	�	w�ǩ�A�* 

Average reward per step۶�� ~�       ��2	 
w�ǩ�A�*

epsilon۶��Z1�d.       ��W�	�x�ǩ�A�* 

Average reward per step۶������       ��2	�x�ǩ�A�*

epsilon۶��"�-.       ��W�	��z�ǩ�A�* 

Average reward per step۶��!Hx�       ��2	��z�ǩ�A�*

epsilon۶�����.       ��W�	��|�ǩ�A�* 

Average reward per step۶���Y��       ��2	��|�ǩ�A�*

epsilon۶����l�.       ��W�	�9�ǩ�A�* 

Average reward per step۶�� ��       ��2	�:�ǩ�A�*

epsilon۶���V?�.       ��W�	<0��ǩ�A�* 

Average reward per step۶��$�D0       ��2	�0��ǩ�A�*

epsilon۶���7U�.       ��W�	2V��ǩ�A�* 

Average reward per step۶��g�K�       ��2	W��ǩ�A�*

epsilon۶���!
'.       ��W�	�v��ǩ�A�* 

Average reward per step۶���I�       ��2	x��ǩ�A�*

epsilon۶����..       ��W�	�j��ǩ�A�* 

Average reward per step۶��䲔@       ��2	vk��ǩ�A�*

epsilon۶���̇.       ��W�	|&��ǩ�A�* 

Average reward per step۶���P�F       ��2	'��ǩ�A�*

epsilon۶�����e.       ��W�	ǂ��ǩ�A�* 

Average reward per step۶��}T�       ��2	f���ǩ�A�*

epsilon۶���#��.       ��W�	G���ǩ�A�* 

Average reward per step۶���ɣ�       ��2	撍�ǩ�A�*

epsilon۶���#�#.       ��W�	m9��ǩ�A�* 

Average reward per step۶���,�       ��2	 :��ǩ�A�*

epsilon۶��(Lt.       ��W�	�n��ǩ�A�* 

Average reward per step۶��
�"�       ��2	io��ǩ�A�*

epsilon۶���B�*.       ��W�	߉��ǩ�A�* 

Average reward per step۶��0��O       ��2	����ǩ�A�*

epsilon۶�����:0       ���_	����ǩ�A2*#
!
Average reward per episode�M�����.       ��W�	����ǩ�A2*!

total reward per episode  ���Q��.       ��W�	S]��ǩ�A�* 

Average reward per step�M�98]       ��2	-^��ǩ�A�*

epsilon�M�����.       ��W�	(~��ǩ�A�* 

Average reward per step�M�.7H       ��2	�~��ǩ�A�*

epsilon�M�Ǘ].       ��W�	k���ǩ�A�* 

Average reward per step�M��*_�       ��2	b���ǩ�A�*

epsilon�M�~e�r.       ��W�	ޝ�ǩ�A�* 

Average reward per step�M��Q       ��2	�ޝ�ǩ�A�*

epsilon�M�;��.       ��W�	D��ǩ�A�* 

Average reward per step�M��xl�       ��2	�D��ǩ�A�*

epsilon�M���.       ��W�	8���ǩ�A�* 

Average reward per step�M�%Wh�       ��2	���ǩ�A�*

epsilon�M�Cؠ.       ��W�	�l��ǩ�A�* 

Average reward per step�M�}+c       ��2	�m��ǩ�A�*

epsilon�M�;�.�.       ��W�	����ǩ�A�* 

Average reward per step�M����       ��2	u���ǩ�A�*

epsilon�M�Z4�6.       ��W�	�Ŵ�ǩ�A�* 

Average reward per step�M�͉       ��2	�ƴ�ǩ�A�*

epsilon�M���i�.       ��W�	q��ǩ�A�* 

Average reward per step�M�8�g       ��2	S��ǩ�A�*

epsilon�M��E��.       ��W�	���ǩ�A�* 

Average reward per step�M�{��       ��2	9��ǩ�A�*

epsilon�M����.       ��W�	6��ǩ�A�* 

Average reward per step�M�BU�f       ��2	8��ǩ�A�*

epsilon�M���29.       ��W�	n���ǩ�A�* 

Average reward per step�M�m�NX       ��2	'���ǩ�A�*

epsilon�M��0rh.       ��W�	�&��ǩ�A�* 

Average reward per step�M�Sbq�       ��2	W'��ǩ�A�*

epsilon�M��ex�.       ��W�	�H��ǩ�A�* 

Average reward per step�M���u�       ��2	�I��ǩ�A�*

epsilon�M���'.       ��W�	r���ǩ�A�* 

Average reward per step�M��
h       ��2	D���ǩ�A�*

epsilon�M�<�R�.       ��W�	�/��ǩ�A�* 

Average reward per step�M�&��       ��2	�0��ǩ�A�*

epsilon�M��\��.       ��W�	�v��ǩ�A�* 

Average reward per step�M��(x       ��2	�w��ǩ�A�*

epsilon�M����.       ��W�	+���ǩ�A�* 

Average reward per step�M�r���       ��2	���ǩ�A�*

epsilon�M�Y�R�.       ��W�	x���ǩ�A�* 

Average reward per step�M���$`       ��2	E���ǩ�A�*

epsilon�M�����.       ��W�	����ǩ�A�* 

Average reward per step�M�����       ��2	����ǩ�A�*

epsilon�M�Z#h.       ��W�	2��ǩ�A�* 

Average reward per step�M�9*FL       ��2	�2��ǩ�A�*

epsilon�M�1�n.       ��W�	R���ǩ�A�* 

Average reward per step�M��7�       ��2	,���ǩ�A�*

epsilon�M�_�.       ��W�	���ǩ�A�* 

Average reward per step�M���[�       ��2	U��ǩ�A�*

epsilon�M��M�d.       ��W�	�-��ǩ�A�* 

Average reward per step�M��̀!       ��2	.��ǩ�A�*

epsilon�M��D(�.       ��W�	D6��ǩ�A�* 

Average reward per step�M�_��d       ��2	�7��ǩ�A�*

epsilon�M���l.       ��W�	�}��ǩ�A�* 

Average reward per step�M��       ��2	A~��ǩ�A�*

epsilon�M�Ϛ8�.       ��W�	2��ǩ�A�* 

Average reward per step�M�� o�       ��2	��ǩ�A�*

epsilon�M�T�H�.       ��W�	���ǩ�A�* 

Average reward per step�M�3�(�       ��2	���ǩ�A�*

epsilon�M����0       ���_	o/��ǩ�A3*#
!
Average reward per episode�ӈ���5,.       ��W�	0��ǩ�A3*!

total reward per episode  ���.       ��W�	M���ǩ�A�* 

Average reward per step�ӈ�Bt\       ��2	4���ǩ�A�*

epsilon�ӈ���g�.       ��W�	�`��ǩ�A�* 

Average reward per step�ӈ��\�       ��2	[a��ǩ�A�*

epsilon�ӈ����Z.       ��W�	.���ǩ�A�* 

Average reward per step�ӈ�'��u       ��2	���ǩ�A�*

epsilon�ӈ��U�.       ��W�	�x��ǩ�A�* 

Average reward per step�ӈ�<;o       ��2	[y��ǩ�A�*

epsilon�ӈ��-s�.       ��W�	$+��ǩ�A�* 

Average reward per step�ӈ�vhm       ��2	,��ǩ�A�*

epsilon�ӈ���ǹ.       ��W�	���ǩ�A�* 

Average reward per step�ӈ��       ��2	����ǩ�A�*

epsilon�ӈ��Nhp.       ��W�	���ǩ�A�* 

Average reward per step�ӈ����       ��2	����ǩ�A�*

epsilon�ӈ��:k9.       ��W�	���ǩ�A�* 

Average reward per step�ӈ�b9��       ��2	����ǩ�A�*

epsilon�ӈ�!�Z.       ��W�	-���ǩ�A�* 

Average reward per step�ӈ�	f"�       ��2	)���ǩ�A�*

epsilon�ӈ��G��.       ��W�	����ǩ�A�* 

Average reward per step�ӈ����n       ��2	����ǩ�A�*

epsilon�ӈ�h~��.       ��W�	���ǩ�A�* 

Average reward per step�ӈ���G*       ��2	l��ǩ�A�*

epsilon�ӈ�>I5	.       ��W�	�ǩ�A�* 

Average reward per step�ӈ��|n�       ��2	��ǩ�A�*

epsilon�ӈ�f��.       ��W�	'�ǩ�A�* 

Average reward per step�ӈ��.R       ��2	��ǩ�A�*

epsilon�ӈ��}.       ��W�	�.�ǩ�A�* 

Average reward per step�ӈ�|,��       ��2	8/�ǩ�A�*

epsilon�ӈ�͆l.       ��W�	YR
�ǩ�A�* 

Average reward per step�ӈ�K��       ��2	�R
�ǩ�A�*

epsilon�ӈ��>�#.       ��W�	���ǩ�A�* 

Average reward per step�ӈ���       ��2	���ǩ�A�*

epsilon�ӈ��cn:.       ��W�	V�ǩ�A�* 

Average reward per step�ӈ� #R�       ��2	�V�ǩ�A�*

epsilon�ӈ���@.       ��W�	�z�ǩ�A�* 

Average reward per step�ӈ�+��       ��2	�{�ǩ�A�*

epsilon�ӈ�#V��.       ��W�	��ǩ�A�* 

Average reward per step�ӈ��Q�8       ��2	w�ǩ�A�*

epsilon�ӈ���0�.       ��W�	�;�ǩ�A�* 

Average reward per step�ӈ�Au��       ��2	><�ǩ�A�*

epsilon�ӈ�S_\0       ���_	�U�ǩ�A4*#
!
Average reward per episode����p|U.       ��W�	V�ǩ�A4*!

total reward per episode  �ª��.       ��W�	���ǩ�A�* 

Average reward per step����c_J       ��2	��ǩ�A�*

epsilon����EA�.       ��W�	ǻ�ǩ�A�* 

Average reward per step������z       ��2	���ǩ�A�*

epsilon�����?�.       ��W�	B�ǩ�A�* 

Average reward per step����;ZLK       ��2	�B�ǩ�A�*

epsilon��������.       ��W�	,�ǩ�A�* 

Average reward per step�����E�       ��2	4-�ǩ�A�*

epsilon�����r�.       ��W�	?t �ǩ�A�* 

Average reward per step������W       ��2	Su �ǩ�A�*

epsilon����#�[l.       ��W�	��$�ǩ�A�* 

Average reward per step�������E       ��2	w�$�ǩ�A�*

epsilon����}.       ��W�	7R(�ǩ�A�* 

Average reward per step����'�h<       ��2	HS(�ǩ�A�*

epsilon�����XS�.       ��W�	��*�ǩ�A�* 

Average reward per step�������       ��2	��*�ǩ�A�*

epsilon����7]�.       ��W�	�.�ǩ�A�* 

Average reward per step������j       ��2	ץ.�ǩ�A�*

epsilon����]&}�.       ��W�	�0�ǩ�A�* 

Average reward per step���� � �       ��2	˼0�ǩ�A�*

epsilon����7�7�.       ��W�	��2�ǩ�A�* 

Average reward per step����}�       ��2	��2�ǩ�A�*

epsilon������BQ.       ��W�	R4�ǩ�A�* 

Average reward per step����|d       ��2	�R4�ǩ�A�*

epsilon����9�C.       ��W�	�n6�ǩ�A�* 

Average reward per step�����}{.       ��2	�o6�ǩ�A�*

epsilon����1��.       ��W�	��8�ǩ�A�* 

Average reward per step�����N�        ��2	�8�ǩ�A�*

epsilon�����"[�.       ��W�	^�:�ǩ�A�* 

Average reward per step�����       ��2	�:�ǩ�A�*

epsilon����v�?�.       ��W�	
�<�ǩ�A�* 

Average reward per step����E��8       ��2	��<�ǩ�A�*

epsilon����E�:�.       ��W�	q ?�ǩ�A�* 

Average reward per step����}x�       ��2	K?�ǩ�A�*

epsilon����Rx�.       ��W�	6A�ǩ�A�* 

Average reward per step�����N       ��2	�A�ǩ�A�*

epsilon����s���.       ��W�	t�B�ǩ�A�* 

Average reward per step�����       ��2	�B�ǩ�A�*

epsilon������;�.       ��W�	��D�ǩ�A�* 

Average reward per step����ő��       ��2	��D�ǩ�A�*

epsilon����,2�.       ��W�	�#G�ǩ�A�* 

Average reward per step����xUӱ       ��2	_$G�ǩ�A�*

epsilon������.       ��W�	�I�ǩ�A�* 

Average reward per step������W,       ��2	\I�ǩ�A�*

epsilon�������.       ��W�	�FK�ǩ�A�* 

Average reward per step����}�*�       ��2	�GK�ǩ�A�*

epsilon����l�w@.       ��W�	f�M�ǩ�A�* 

Average reward per step������H�       ��2	]�M�ǩ�A�*

epsilon����<�F�.       ��W�	RDO�ǩ�A�* 

Average reward per step������       ��2	5EO�ǩ�A�*

epsilon����.�~.       ��W�	��Q�ǩ�A�* 

Average reward per step������0
       ��2	��Q�ǩ�A�*

epsilon����W�.       ��W�	7�U�ǩ�A�* 

Average reward per step�����B�       ��2	ҋU�ǩ�A�*

epsilon�����檻.       ��W�	�\�ǩ�A�* 

Average reward per step�����A       ��2	��\�ǩ�A�*

epsilon����k���.       ��W�	MI_�ǩ�A�* 

Average reward per step�������       ��2	IJ_�ǩ�A�*

epsilon������jw.       ��W�	�c�ǩ�A�* 

Average reward per step������ �       ��2	� c�ǩ�A�*

epsilon����K�.       ��W�	��d�ǩ�A�* 

Average reward per step�����$�]       ��2	ɬd�ǩ�A�*

epsilon������-�.       ��W�	��f�ǩ�A�* 

Average reward per step����͏�*       ��2	��f�ǩ�A�*

epsilon����u%�.       ��W�	�6i�ǩ�A�* 

Average reward per step����˭��       ��2	�7i�ǩ�A�*

epsilon����>GO.       ��W�	Ւk�ǩ�A�* 

Average reward per step����}��       ��2	��k�ǩ�A�*

epsilon�����%[�.       ��W�	am�ǩ�A�* 

Average reward per step�����(       ��2	(bm�ǩ�A�*

epsilon����M��.       ��W�	1�o�ǩ�A�* 

Average reward per step������d$       ��2	)�o�ǩ�A�*

epsilon����F㔊.       ��W�	5�q�ǩ�A�* 

Average reward per step����M       ��2	�q�ǩ�A�*

epsilon�������.       ��W�	�cs�ǩ�A�* 

Average reward per step����f�x       ��2	�ds�ǩ�A�*

epsilon����.       ��W�	v�u�ǩ�A�* 

Average reward per step����5^v�       ��2	3�u�ǩ�A�*

epsilon�����x.       ��W�	!�w�ǩ�A�* 

Average reward per step����/�@�       ��2	��w�ǩ�A�*

epsilon�������&.       ��W�	�oy�ǩ�A�* 

Average reward per step������       ��2	qy�ǩ�A�*

epsilon����ͧ R.       ��W�	 �{�ǩ�A�* 

Average reward per step��������       ��2	��{�ǩ�A�*

epsilon�����?bI.       ��W�	�B~�ǩ�A�* 

Average reward per step�����Q�
       ��2	�C~�ǩ�A�*

epsilon������,�.       ��W�	����ǩ�A�* 

Average reward per step����9FH       ��2	���ǩ�A�*

epsilon������=.       ��W�	|C��ǩ�A�* 

Average reward per step�����z�       ��2	D��ǩ�A�*

epsilon����J^��0       ���_	Ec��ǩ�A5*#
!
Average reward per episode?�� h�i.       ��W�	d��ǩ�A5*!

total reward per episode  ��4�	.       ��W�	U���ǩ�A�* 

Average reward per step?��N{�       ��2	@���ǩ�A�*

epsilon?��Mtk.       ��W�	 z��ǩ�A�* 

Average reward per step?����       ��2	{��ǩ�A�*

epsilon?���f'.       ��W�	ٯ��ǩ�A�* 

Average reward per step?�����%       ��2	����ǩ�A�*

epsilon?���x.       ��W�	+���ǩ�A�* 

Average reward per step?����E�       ��2	ƥ��ǩ�A�*

epsilon?��J�b.       ��W�	 ���ǩ�A�* 

Average reward per step?��'ā�       ��2	����ǩ�A�*

epsilon?��j�8.       ��W�	]���ǩ�A�* 

Average reward per step?�����       ��2	8���ǩ�A�*

epsilon?����w.       ��W�	�Q��ǩ�A�* 

Average reward per step?��!�Y       ��2	�R��ǩ�A�*

epsilon?���S�.       ��W�	� ��ǩ�A�* 

Average reward per step?���A�       ��2	�!��ǩ�A�*

epsilon?�����y.       ��W�		���ǩ�A�* 

Average reward per step?�����       ��2	����ǩ�A�*

epsilon?��,��[.       ��W�	����ǩ�A�* 

Average reward per step?����        ��2	R���ǩ�A�*

epsilon?����߆.       ��W�	鸟�ǩ�A�* 

Average reward per step?���s��       ��2	����ǩ�A�*

epsilon?�����.       ��W�	U1��ǩ�A�* 

Average reward per step?��cE�       ��2	�1��ǩ�A�*

epsilon?���6.       ��W�	IG��ǩ�A�* 

Average reward per step?��o�R_       ��2	H��ǩ�A�*

epsilon?��	Њ.       ��W�	Z��ǩ�A�* 

Average reward per step?����g�       ��2	�Z��ǩ�A�*

epsilon?��h��.       ��W�	�w��ǩ�A�* 

Average reward per step?�����D       ��2	�x��ǩ�A�*

epsilon?��2�U�.       ��W�	����ǩ�A�* 

Average reward per step?����V�       ��2	���ǩ�A�*

epsilon?���BYB.       ��W�	4���ǩ�A�* 

Average reward per step?��h�       ��2	圫�ǩ�A�*

epsilon?��'Ӣ.       ��W�	\���ǩ�A�* 

Average reward per step?��O�5       ��2	���ǩ�A�*

epsilon?��ř�0       ���_	ɭ�ǩ�A6*#
!
Average reward per episode�	����.       ��W�	�ɭ�ǩ�A6*!

total reward per episode  �a��.       ��W�	2!��ǩ�A�* 

Average reward per step�	�SUp�       ��2	"��ǩ�A�*

epsilon�	�b�.       ��W�	u���ǩ�A�* 

Average reward per step�	�1��       ��2	G���ǩ�A�*

epsilon�	�d��..       ��W�	i���ǩ�A�* 

Average reward per step�	�@�ԥ       ��2	���ǩ�A�*

epsilon�	�0��.       ��W�	~���ǩ�A�* 

Average reward per step�	�/��       ��2	����ǩ�A�*

epsilon�	�\��.       ��W�	�Q��ǩ�A�* 

Average reward per step�	��ME�       ��2	aR��ǩ�A�*

epsilon�	�W�H�.       ��W�	�X��ǩ�A�* 

Average reward per step�	�4��D       ��2	qY��ǩ�A�*

epsilon�	�,�f.       ��W�	G8��ǩ�A�* 

Average reward per step�	��EIh       ��2	�8��ǩ�A�*

epsilon�	�&k��.       ��W�	M��ǩ�A�* 

Average reward per step�	�4�       ��2	�M��ǩ�A�*

epsilon�	�O���.       ��W�	����ǩ�A�* 

Average reward per step�	���       ��2	����ǩ�A�*

epsilon�	��#3�.       ��W�	����ǩ�A�* 

Average reward per step�	���ݮ       ��2	����ǩ�A�*

epsilon�	�lSF.       ��W�	D��ǩ�A�* 

Average reward per step�	��4!�       ��2	�D��ǩ�A�*

epsilon�	�Kd�z.       ��W�	E���ǩ�A�* 

Average reward per step�	��8�       ��2	���ǩ�A�*

epsilon�	�D��.       ��W�	J���ǩ�A�* 

Average reward per step�	�m�Հ       ��2	���ǩ�A�*

epsilon�	���p�.       ��W�	0���ǩ�A�* 

Average reward per step�	��>�       ��2	+���ǩ�A�*

epsilon�	�+�[�.       ��W�	����ǩ�A�* 

Average reward per step�	�c�       ��2	H���ǩ�A�*

epsilon�	�'��.       ��W�	H���ǩ�A�* 

Average reward per step�	��n�       ��2	"���ǩ�A�*

epsilon�	���.       ��W�	�(��ǩ�A�* 

Average reward per step�	�P��5       ��2	�)��ǩ�A�*

epsilon�	��Z��.       ��W�	£��ǩ�A�* 

Average reward per step�	�X�$�       ��2	a���ǩ�A�*

epsilon�	����I.       ��W�	N���ǩ�A�* 

Average reward per step�	�Igq#       ��2	5���ǩ�A�*

epsilon�	�I��.       ��W�	<���ǩ�A�* 

Average reward per step�	�X�<�       ��2	���ǩ�A�*

epsilon�	�xJ.       ��W�	���ǩ�A�* 

Average reward per step�	���_       ��2	���ǩ�A�*

epsilon�	����^.       ��W�	5@��ǩ�A�* 

Average reward per step�	���       ��2	�@��ǩ�A�*

epsilon�	��S�.       ��W�	fL��ǩ�A�* 

Average reward per step�	��f�C       ��2	<M��ǩ�A�*

epsilon�	�����.       ��W�	.r��ǩ�A�* 

Average reward per step�	�"�٨       ��2	s��ǩ�A�*

epsilon�	����T.       ��W�	X��ǩ�A�* 

Average reward per step�	�y��;       ��2	T��ǩ�A�*

epsilon�	�_<��0       ���_	y��ǩ�A7*#
!
Average reward per episode\���;vu�.       ��W�	% ��ǩ�A7*!

total reward per episode  ��rj.       ��W�	w���ǩ�A�* 

Average reward per step\���߿ȭ       ��2	^���ǩ�A�*

epsilon\���і݀.       ��W�	6��ǩ�A�* 

Average reward per step\���P͑�       ��2	�6��ǩ�A�*

epsilon\���ha1J.       ��W�	s���ǩ�A�* 

Average reward per step\���9i�       ��2	@���ǩ�A�*

epsilon\������t.       ��W�	L���ǩ�A�* 

Average reward per step\����:��       ��2	����ǩ�A�*

epsilon\�����.       ��W�	+���ǩ�A�* 

Average reward per step\���L�4�       ��2	Y���ǩ�A�*

epsilon\�����7.       ��W�	���ǩ�A�* 

Average reward per step\����7�       ��2	����ǩ�A�*

epsilon\������.       ��W�	����ǩ�A�* 

Average reward per step\���}�w       ��2	c���ǩ�A�*

epsilon\���l'�j.       ��W�	����ǩ�A�* 

Average reward per step\����<       ��2	� ��ǩ�A�*

epsilon\�����PM.       ��W�	L��ǩ�A�* 

Average reward per step\�����j:       ��2	���ǩ�A�*

epsilon\������.       ��W�	���ǩ�A�* 

Average reward per step\����"��       ��2	����ǩ�A�*

epsilon\���_(�`.       ��W�	Q��ǩ�A�* 

Average reward per step\����E6�       ��2	���ǩ�A�*

epsilon\���녛.       ��W�	.Y�ǩ�A�* 

Average reward per step\���67�D       ��2	Z�ǩ�A�*

epsilon\������.       ��W�	��ǩ�A�* 

Average reward per step\�������       ��2	���ǩ�A�*

epsilon\���8���.       ��W�	-A�ǩ�A�* 

Average reward per step\����5�0       ��2	�A�ǩ�A�*

epsilon\�������.       ��W�	:��ǩ�A�* 

Average reward per step\����d��       ��2	ڒ�ǩ�A�*

epsilon\���'��e.       ��W�	"	�ǩ�A�* 

Average reward per step\���Ol��       ��2	�"	�ǩ�A�*

epsilon\���\'y.       ��W�	���ǩ�A�* 

Average reward per step\���:��       ��2	2��ǩ�A�*

epsilon\����s�.       ��W�	n��ǩ�A�* 

Average reward per step\������       ��2	D��ǩ�A�*

epsilon\����I�1.       ��W�	���ǩ�A�* 

Average reward per step\���A��?       ��2	~ �ǩ�A�*

epsilon\���Bθ.       ��W�	8��ǩ�A�* 

Average reward per step\���7�}>       ��2	��ǩ�A�*

epsilon\����I[�.       ��W�	���ǩ�A�* 

Average reward per step\���3]�C       ��2	���ǩ�A�*

epsilon\����Ѵ$.       ��W�	���ǩ�A�* 

Average reward per step\����<��       ��2	���ǩ�A�*

epsilon\����;?q.       ��W�	��ǩ�A�* 

Average reward per step\���DQ"       ��2	4�ǩ�A�*

epsilon\���7��M.       ��W�	;��ǩ�A�* 

Average reward per step\���,r�F       ��2	 ��ǩ�A�*

epsilon\����iY�.       ��W�	���ǩ�A�* 

Average reward per step\����R��       ��2	���ǩ�A�*

epsilon\���Y5،.       ��W�	�D�ǩ�A�* 

Average reward per step\����i       ��2	�E�ǩ�A�*

epsilon\������[.       ��W�	d �ǩ�A�* 

Average reward per step\���V��W       ��2	B �ǩ�A�*

epsilon\����|L�.       ��W�	�}"�ǩ�A�* 

Average reward per step\�����0�       ��2	�~"�ǩ�A�*

epsilon\���\���.       ��W�	�$�ǩ�A�* 

Average reward per step\����Ê�       ��2	�$�ǩ�A�*

epsilon\����ouQ.       ��W�	�(&�ǩ�A�* 

Average reward per step\����a(       ��2	V)&�ǩ�A�*

epsilon\���D�b�.       ��W�	�A(�ǩ�A�* 

Average reward per step\����3h       ��2	C(�ǩ�A�*

epsilon\�����7.       ��W�	g*�ǩ�A�* 

Average reward per step\���G�6       ��2	�g*�ǩ�A�*

epsilon\���,��'.       ��W�	x{,�ǩ�A�* 

Average reward per step\����XS�       ��2	N|,�ǩ�A�*

epsilon\���"Y�.       ��W�	��-�ǩ�A�* 

Average reward per step\���&��;       ��2	��-�ǩ�A�*

epsilon\������.       ��W�	=d/�ǩ�A�* 

Average reward per step\�����i       ��2	�d/�ǩ�A�*

epsilon\���	�a�.       ��W�	{1�ǩ�A�* 

Average reward per step\���*�ex       ��2	1�ǩ�A�*

epsilon\���!��e.       ��W�	�<3�ǩ�A�* 

Average reward per step\���n@FV       ��2	�=3�ǩ�A�*

epsilon\�������.       ��W�	�q5�ǩ�A�* 

Average reward per step\����KR       ��2	Tr5�ǩ�A�*

epsilon\���3���.       ��W�	I�7�ǩ�A�* 

Average reward per step\�����]       ��2	��7�ǩ�A�*

epsilon\���e��'.       ��W�	|H9�ǩ�A�* 

Average reward per step\���A]�       ��2	RI9�ǩ�A�*

epsilon\�����j.       ��W�	��;�ǩ�A�* 

Average reward per step\�����-|       ��2	b�;�ǩ�A�*

epsilon\�����L�.       ��W�	��=�ǩ�A�* 

Average reward per step\����.�       ��2	��=�ǩ�A�*

epsilon\�����F.       ��W�	��?�ǩ�A�* 

Average reward per step\����X-a       ��2	��?�ǩ�A�*

epsilon\���Ƶ_.       ��W�	�A�ǩ�A�* 

Average reward per step\���H�Zz       ��2	�A�ǩ�A�*

epsilon\����ܸ�.       ��W�	2D�ǩ�A�* 

Average reward per step\�����       ��2	D�ǩ�A�*

epsilon\���~�Y�.       ��W�	*�E�ǩ�A�* 

Average reward per step\����*�6       ��2	��E�ǩ�A�*

epsilon\���D��.       ��W�	q�G�ǩ�A�* 

Average reward per step\����2NR       ��2	` H�ǩ�A�*

epsilon\����n�_.       ��W�	��I�ǩ�A�* 

Average reward per step\�����       ��2	��I�ǩ�A�*

epsilon\���T���0       ���_	��I�ǩ�A8*#
!
Average reward per episode  ��NZ�.       ��W�	y�I�ǩ�A8*!

total reward per episode  ��lZ.       ��W�	yN�ǩ�A�* 

Average reward per step  ��(��       ��2	�yN�ǩ�A�*

epsilon  ���*g�.       ��W�	n1P�ǩ�A�* 

Average reward per step  ��|N�       ��2	2P�ǩ�A�*

epsilon  ����8�.       ��W�	ZcR�ǩ�A�* 

Average reward per step  ��%v       ��2	�cR�ǩ�A�*

epsilon  ��c��.       ��W�	F�T�ǩ�A�* 

Average reward per step  ��ޡ��       ��2	�T�ǩ�A�*

epsilon  ����7.       ��W�	�=V�ǩ�A�* 

Average reward per step  ���iR       ��2	�>V�ǩ�A�*

epsilon  ���$�*.       ��W�	,�X�ǩ�A�* 

Average reward per step  ���%�w       ��2	�X�ǩ�A�*

epsilon  ����.       ��W�	��Z�ǩ�A�* 

Average reward per step  ���[�       ��2	��Z�ǩ�A�*

epsilon  ���\
M.       ��W�	]�ǩ�A�* 

Average reward per step  ���(�}       ��2	�]�ǩ�A�*

epsilon  �� �].       ��W�	�`�ǩ�A�* 

Average reward per step  �����       ��2	)�`�ǩ�A�*

epsilon  ���`�.       ��W�	Z�b�ǩ�A�* 

Average reward per step  �����       ��2	�b�ǩ�A�*

epsilon  ��Bt}p.       ��W�	zd�ǩ�A�* 

Average reward per step  ����X       ��2	�{d�ǩ�A�*

epsilon  ����r>.       ��W�	O�f�ǩ�A�* 

Average reward per step  ��� �       ��2	1�f�ǩ�A�*

epsilon  ��4m�2.       ��W�	x�h�ǩ�A�* 

Average reward per step  ���Epl       ��2	0�h�ǩ�A�*

epsilon  ����1�.       ��W�	�k�ǩ�A�* 

Average reward per step  ��beM       ��2	!k�ǩ�A�*

epsilon  ��])Z�.       ��W�	vm�ǩ�A�* 

Average reward per step  ���x?       ��2	m�ǩ�A�*

epsilon  ��̕��.       ��W�	No�ǩ�A�* 

Average reward per step  ��6>�       ��2	�No�ǩ�A�*

epsilon  ��X��.       ��W�	1}q�ǩ�A�* 

Average reward per step  ����5�       ��2	~q�ǩ�A�*

epsilon  ����f.       ��W�	"�r�ǩ�A�* 

Average reward per step  ���, �       ��2	i�r�ǩ�A�*

epsilon  ������.       ��W�	�@u�ǩ�A�* 

Average reward per step  ���^��       ��2	�Au�ǩ�A�*

epsilon  ��Y�9.       ��W�	q�w�ǩ�A�* 

Average reward per step  ����F       ��2	W�w�ǩ�A�*

epsilon  ���v7.       ��W�	my�ǩ�A�* 

Average reward per step  ���i��       ��2	 y�ǩ�A�*

epsilon  ��cL�.       ��W�	�`{�ǩ�A�* 

Average reward per step  ���v��       ��2	�a{�ǩ�A�*

epsilon  �����.       ��W�	��}�ǩ�A�* 

Average reward per step  ���.�       ��2	u�}�ǩ�A�*

epsilon  ���=�.       ��W�	���ǩ�A�* 

Average reward per step  ��p���       ��2	���ǩ�A�*

epsilon  �����.       ��W�	Yn��ǩ�A�* 

Average reward per step  ��3M       ��2	�o��ǩ�A�*

epsilon  �� �.       ��W�	*��ǩ�A�* 

Average reward per step  ��r�\       ��2	���ǩ�A�*

epsilon  ���O.       ��W�	_΅�ǩ�A�* 

Average reward per step  ���kĺ       ��2	υ�ǩ�A�*

epsilon  ����.       ��W�	W!��ǩ�A�* 

Average reward per step  ��o��+       ��2	"��ǩ�A�*

epsilon  ��$�r�.       ��W�	y��ǩ�A�* 

Average reward per step  ���j�T       ��2	-��ǩ�A�*

epsilon  ���0Y0       ���_	&��ǩ�A9*#
!
Average reward per episode�=���l�D.       ��W�	�&��ǩ�A9*!

total reward per episode   Î��.       ��W�	���ǩ�A�* 

Average reward per step�=���..*       ��2	w���ǩ�A�*

epsilon�=����cj.       ��W�	,b��ǩ�A�* 

Average reward per step�=��S���       ��2	c��ǩ�A�*

epsilon�=����/.       ��W�	>���ǩ�A�* 

Average reward per step�=������       ��2	���ǩ�A�*

epsilon�=���v
[.       ��W�	m��ǩ�A�* 

Average reward per step�=��^*f�       ��2	�m��ǩ�A�*

epsilon�=��[^.       ��W�	Z���ǩ�A�* 

Average reward per step�=��y�+�       ��2	���ǩ�A�*

epsilon�=���N&�.       ��W�	���ǩ�A�* 

Average reward per step�=�� X        ��2	���ǩ�A�*

epsilon�=���p�.       ��W�	�L��ǩ�A�* 

Average reward per step�=����F       ��2	bN��ǩ�A�*

epsilon�=���il.       ��W�	�H��ǩ�A�* 

Average reward per step�=���X*�       ��2	kJ��ǩ�A�*

epsilon�=���8�7.       ��W�	px��ǩ�A�* 

Average reward per step�=��k}�r       ��2	�y��ǩ�A�*

epsilon�=���;L�.       ��W�	pw��ǩ�A�* 

Average reward per step�=��kQ�       ��2	Sx��ǩ�A�*

epsilon�=����i�.       ��W�	wI��ǩ�A�* 

Average reward per step�=����P�       ��2	fJ��ǩ�A�*

epsilon�=�����.       ��W�	�C��ǩ�A�* 

Average reward per step�=���w5       ��2	E��ǩ�A�*

epsilon�=��nR�.       ��W�	����ǩ�A�* 

Average reward per step�=����s�       ��2	��ǩ�A�*

epsilon�=����Zq.       ��W�	���ǩ�A�* 

Average reward per step�=���A       ��2	\��ǩ�A�*

epsilon�=���/�.       ��W�	���ǩ�A�* 

Average reward per step�=���8b       ��2	ǁ��ǩ�A�*

epsilon�=��7���.       ��W�	�ҳ�ǩ�A�* 

Average reward per step�=���8B�       ��2	�ӳ�ǩ�A�*

epsilon�=�����=.       ��W�	���ǩ�A�* 

Average reward per step�=���j�       ��2	���ǩ�A�*

epsilon�=���mG�.       ��W�	����ǩ�A�* 

Average reward per step�=��/d�v       ��2	I���ǩ�A�*

epsilon�=����US.       ��W�	����ǩ�A�* 

Average reward per step�=����X�       ��2	u���ǩ�A�*

epsilon�=���~��.       ��W�	��ǩ�A�* 

Average reward per step�=���L+J       ��2	��ǩ�A�*

epsilon�=��F(�{.       ��W�	�q��ǩ�A�* 

Average reward per step�=���=7W       ��2	s��ǩ�A�*

epsilon�=���n�f.       ��W�	(��ǩ�A�* 

Average reward per step�=��ֲT�       ��2	���ǩ�A�*

epsilon�=���
��.       ��W�	���ǩ�A�* 

Average reward per step�=���Z�#       ��2	���ǩ�A�*

epsilon�=��s��a.       ��W�	$��ǩ�A�* 

Average reward per step�=����~�       ��2	�$��ǩ�A�*

epsilon�=��|�V.       ��W�	=d��ǩ�A�* 

Average reward per step�=��e�x       ��2	#e��ǩ�A�*

epsilon�=���뉘0       ���_	����ǩ�A:*#
!
Average reward per episode�p���M��.       ��W�	b���ǩ�A:*!

total reward per episode  �=��.       ��W�	lA��ǩ�A�* 

Average reward per step�p���HOF       ��2	$B��ǩ�A�*

epsilon�p����G..       ��W�	�r��ǩ�A�* 

Average reward per step�p��k�u       ��2	�s��ǩ�A�*

epsilon�p��U�0�.       ��W�	p"��ǩ�A�* 

Average reward per step�p��Fw�|       ��2	�#��ǩ�A�*

epsilon�p����My.       ��W�	�{��ǩ�A�* 

Average reward per step�p���!��       ��2	�|��ǩ�A�*

epsilon�p��a�o�.       ��W�	���ǩ�A�* 

Average reward per step�p���g�       ��2	���ǩ�A�*

epsilon�p��4���.       ��W�	�G��ǩ�A�* 

Average reward per step�p����F_       ��2	�H��ǩ�A�*

epsilon�p�����.       ��W�	/l��ǩ�A�* 

Average reward per step�p����h       ��2	m��ǩ�A�*

epsilon�p��u��.       ��W�	����ǩ�A�* 

Average reward per step�p��a�h�       ��2	{���ǩ�A�*

epsilon�p�����.       ��W�	eq��ǩ�A�* 

Average reward per step�p��Aql       ��2	*r��ǩ�A�*

epsilon�p�����.       ��W�	���ǩ�A�* 

Average reward per step�p����3       ��2	����ǩ�A�*

epsilon�p����>�.       ��W�	#2��ǩ�A�* 

Average reward per step�p���w�       ��2	H3��ǩ�A�*

epsilon�p��cR46.       ��W�	���ǩ�A�* 

Average reward per step�p���W       ��2	f��ǩ�A�*

epsilon�p����_.       ��W�	����ǩ�A�* 

Average reward per step�p���'E�       ��2	����ǩ�A�*

epsilon�p���!�*.       ��W�	�y��ǩ�A�* 

Average reward per step�p�����@       ��2	�z��ǩ�A�*

epsilon�p�����o.       ��W�	����ǩ�A�* 

Average reward per step�p����:}       ��2	���ǩ�A�*

epsilon�p��Ӡ��.       ��W�	����ǩ�A�* 

Average reward per step�p������       ��2	����ǩ�A�*

epsilon�p��T��.       ��W�	o��ǩ�A�* 

Average reward per step�p��)7҄       ��2	&p��ǩ�A�*

epsilon�p�����;.       ��W�	�]��ǩ�A�* 

Average reward per step�p����ɼ       ��2	5_��ǩ�A�*

epsilon�p��c8$.       ��W�	���ǩ�A�* 

Average reward per step�p����       ��2	����ǩ�A�*

epsilon�p��꡻".       ��W�	=) �ǩ�A�* 

Average reward per step�p��0@       ��2	* �ǩ�A�*

epsilon�p��#��.       ��W�	c^�ǩ�A�* 

Average reward per step�p����O       ��2	)_�ǩ�A�*

epsilon�p��q(�n.       ��W�	���ǩ�A�* 

Average reward per step�p����?�       ��2	���ǩ�A�*

epsilon�p��#O.       ��W�	�D�ǩ�A�* 

Average reward per step�p��~/)Z       ��2	�E�ǩ�A�*

epsilon�p��F�qY.       ��W�	���ǩ�A�* 

Average reward per step�p�����u       ��2	'��ǩ�A�*

epsilon�p��}mG�0       ���_	���ǩ�A;*#
!
Average reward per episode  ���_y.       ��W�	P��ǩ�A;*!

total reward per episode  %ñ{.       ��W�	���ǩ�A�* 

Average reward per step  ���@T        ��2	G��ǩ�A�*

epsilon  ��a���.       ��W�	9`�ǩ�A�* 

Average reward per step  ��jlF       ��2	�`�ǩ�A�*

epsilon  ��2��I.       ��W�	���ǩ�A�* 

Average reward per step  ��ZP3~       ��2	X��ǩ�A�*

epsilon  ���YO.       ��W�	���ǩ�A�* 

Average reward per step  �����L       ��2	0��ǩ�A�*

epsilon  ��k�i.       ��W�	Z��ǩ�A�* 

Average reward per step  ��di�       ��2	��ǩ�A�*

epsilon  ��*Y>.       ��W�	t��ǩ�A�* 

Average reward per step  ���}�)       ��2	N��ǩ�A�*

epsilon  ��ޮt.       ��W�	k��ǩ�A�* 

Average reward per step  ������       ��2	��ǩ�A�*

epsilon  ��!��.       ��W�	-��ǩ�A�* 

Average reward per step  ����       ��2	���ǩ�A�*

epsilon  ����'.       ��W�	�F�ǩ�A�* 

Average reward per step  ���y�       ��2	�G�ǩ�A�*

epsilon  ��̮.�.       ��W�	�8!�ǩ�A�* 

Average reward per step  ����HT       ��2	u9!�ǩ�A�*

epsilon  ����	u.       ��W�	�n#�ǩ�A�* 

Average reward per step  ���4�G       ��2	Hp#�ǩ�A�*

epsilon  ���Ӆ�.       ��W�	��%�ǩ�A�* 

Average reward per step  ��W�Za       ��2	Ӿ%�ǩ�A�*

epsilon  ����%.       ��W�	�R'�ǩ�A�* 

Average reward per step  ��2�c�       ��2	�S'�ǩ�A�*

epsilon  ���ٻ�.       ��W�	�})�ǩ�A�* 

Average reward per step  ��Q"�       ��2	�~)�ǩ�A�*

epsilon  ���_.       ��W�	��+�ǩ�A�* 

Average reward per step  ��U��?       ��2	��+�ǩ�A�*

epsilon  ��䆄.       ��W�	G�-�ǩ�A�* 

Average reward per step  ����       ��2	;�-�ǩ�A�*

epsilon  ����.       ��W�	&0�ǩ�A�* 

Average reward per step  ��*`�       ��2	T0�ǩ�A�*

epsilon  ����T.       ��W�	d4�ǩ�A�* 

Average reward per step  ���$�0       ��2	e4�ǩ�A�*

epsilon  ���;�.       ��W�	�8�ǩ�A�* 

Average reward per step  ����d       ��2	�8�ǩ�A�*

epsilon  ��w)�.       ��W�	�<�ǩ�A�* 

Average reward per step  ��0G&I       ��2	c�<�ǩ�A�*

epsilon  ���Xe.       ��W�	.q@�ǩ�A�* 

Average reward per step  ����T       ��2	r@�ǩ�A�*

epsilon  ���ֵ2.       ��W�	��B�ǩ�A�* 

Average reward per step  ����)o       ��2	N�B�ǩ�A�*

epsilon  ��w�R�.       ��W�	2rD�ǩ�A�* 

Average reward per step  ���ġ�       ��2	�rD�ǩ�A�*

epsilon  ���CP.       ��W�	�F�ǩ�A�* 

Average reward per step  ����M       ��2	ٔF�ǩ�A�*

epsilon  �����.       ��W�	�H�ǩ�A�* 

Average reward per step  ���mY0       ��2	ػH�ǩ�A�*

epsilon  �����.       ��W�	JJ�ǩ�A�* 

Average reward per step  ������       ��2	�JJ�ǩ�A�*

epsilon  ����\�.       ��W�	�lL�ǩ�A�* 

Average reward per step  ����Zy       ��2	nmL�ǩ�A�*

epsilon  ���_c�.       ��W�	��N�ǩ�A�* 

Average reward per step  �� ��       ��2	8�N�ǩ�A�*

epsilon  ��}d3�0       ���_	��N�ǩ�A<*#
!
Average reward per episode�m���V�.       ��W�	�N�ǩ�A<*!

total reward per episode  ��.       ��W�	x�R�ǩ�A�* 

Average reward per step�m��#�g       ��2	=�R�ǩ�A�*

epsilon�m������.       ��W�	�T�ǩ�A�* 

Average reward per step�m��}f��       ��2	��T�ǩ�A�*

epsilon�m��r���.       ��W�	�{W�ǩ�A�* 

Average reward per step�m���X+�       ��2	�|W�ǩ�A�*

epsilon�m��Ӎ5�.       ��W�	��X�ǩ�A�* 

Average reward per step�m������       ��2	��X�ǩ�A�*

epsilon�m��61C.       ��W�	�[�ǩ�A�* 

Average reward per step�m����       ��2	��[�ǩ�A�*

epsilon�m���9=�.       ��W�	j�_�ǩ�A�* 

Average reward per step�m��ٽ~       ��2	M�_�ǩ�A�*

epsilon�m��5L�.       ��W�	e�c�ǩ�A�* 

Average reward per step�m��+NC       ��2	T�c�ǩ�A�*

epsilon�m��zn��.       ��W�	2�g�ǩ�A�* 

Average reward per step�m��9Ѳ       ��2	�g�ǩ�A�*

epsilon�m��vw�.       ��W�	�l�ǩ�A�* 

Average reward per step�m������       ��2	�l�ǩ�A�*

epsilon�m���l3}.       ��W�	:�m�ǩ�A�* 

Average reward per step�m���:k;       ��2	1�m�ǩ�A�*

epsilon�m���"L.       ��W�	:#p�ǩ�A�* 

Average reward per step�m����       ��2	�#p�ǩ�A�*

epsilon�m���j��.       ��W�	��q�ǩ�A�* 

Average reward per step�m���>t       ��2	ߤq�ǩ�A�*

epsilon�m���	=�.       ��W�	��s�ǩ�A�* 

Average reward per step�m�����+       ��2	��s�ǩ�A�*

epsilon�m��p��4.       ��W�	��u�ǩ�A�* 

Average reward per step�m���ȼ�       ��2	��u�ǩ�A�*

epsilon�m��>n$-.       ��W�	�/x�ǩ�A�* 

Average reward per step�m��j��8       ��2	{0x�ǩ�A�*

epsilon�m��tm�h.       ��W�	ݵy�ǩ�A�* 

Average reward per step�m��M}�       ��2	̶y�ǩ�A�*

epsilon�m����c�.       ��W�	��{�ǩ�A�* 

Average reward per step�m��=)       ��2	��{�ǩ�A�*

epsilon�m��#���.       ��W�	C~�ǩ�A�* 

Average reward per step�m��oz�       ��2	!~�ǩ�A�*

epsilon�m��u�Xn.       ��W�	�u��ǩ�A�* 

Average reward per step�m��	��2       ��2	�v��ǩ�A�*

epsilon�m���`�J.       ��W�	iÃ�ǩ�A�* 

Average reward per step�m���]�       ��2	Pă�ǩ�A�*

epsilon�m��t���.       ��W�	����ǩ�A�* 

Average reward per step�m��C �       ��2	����ǩ�A�*

epsilon�m��.�g,.       ��W�	ÿ�ǩ�A�* 

Average reward per step�m���F��       ��2	S͈�ǩ�A�*

epsilon�m����:.       ��W�	���ǩ�A�* 

Average reward per step�m��D�S�       ��2	���ǩ�A�*

epsilon�m��$v�y.       ��W�	�B��ǩ�A�* 

Average reward per step�m��7���       ��2	pC��ǩ�A�*

epsilon�m��2��.       ��W�	n���ǩ�A�* 

Average reward per step�m���![       ��2	e���ǩ�A�*

epsilon�m���v�_.       ��W�	����ǩ�A�* 

Average reward per step�m���x��       ��2	j���ǩ�A�*

epsilon�m����8S.       ��W�	�*��ǩ�A�* 

Average reward per step�m��@���       ��2	�+��ǩ�A�*

epsilon�m��Q.       ��W�	6v��ǩ�A�* 

Average reward per step�m����j�       ��2	�v��ǩ�A�*

epsilon�m��O4��.       ��W�	����ǩ�A�* 

Average reward per step�m��6 @r       ��2	e���ǩ�A�*

epsilon�m��`���.       ��W�		���ǩ�A�* 

Average reward per step�m������       ��2	����ǩ�A�*

epsilon�m���P�.       ��W�	"���ǩ�A�* 

Average reward per step�m��'��       ��2	Ō��ǩ�A�*

epsilon�m��"�@'.       ��W�	u���ǩ�A�* 

Average reward per step�m��e>��       ��2	���ǩ�A�*

epsilon�m����.       ��W�	L���ǩ�A�* 

Average reward per step�m����W�       ��2	����ǩ�A�*

epsilon�m��IXW.       ��W�	v���ǩ�A�* 

Average reward per step�m��*�4       ��2	m���ǩ�A�*

epsilon�m���F�.       ��W�	����ǩ�A�* 

Average reward per step�m��.��!       ��2	G���ǩ�A�*

epsilon�m����.       ��W�	�@��ǩ�A�* 

Average reward per step�m���w�       ��2	�A��ǩ�A�*

epsilon�m��f,].       ��W�	V��ǩ�A�* 

Average reward per step�m����8C       ��2	���ǩ�A�*

epsilon�m���q0.       ��W�	�_��ǩ�A�* 

Average reward per step�m�����       ��2	�`��ǩ�A�*

epsilon�m���RH.       ��W�	�N��ǩ�A�* 

Average reward per step�m��j���       ��2	HO��ǩ�A�*

epsilon�m��- �.       ��W�	�1��ǩ�A�* 

Average reward per step�m�����Q       ��2	�2��ǩ�A�*

epsilon�m��s�_.       ��W�	�۳�ǩ�A�* 

Average reward per step�m����N�       ��2	�ܳ�ǩ�A�*

epsilon�m��5�"�.       ��W�	���ǩ�A�* 

Average reward per step�m����e       ��2	����ǩ�A�*

epsilon�m����.       ��W�	���ǩ�A�* 

Average reward per step�m���&�       ��2	���ǩ�A�*

epsilon�m��eYͨ.       ��W�	����ǩ�A�* 

Average reward per step�m���pp�       ��2	����ǩ�A�*

epsilon�m��
�~�.       ��W�	���ǩ�A�* 

Average reward per step�m���UQO       ��2	���ǩ�A�*

epsilon�m��GX��.       ��W�	�9��ǩ�A�* 

Average reward per step�m���nq�       ��2	?:��ǩ�A�*

epsilon�m������.       ��W�	���ǩ�A�* 

Average reward per step�m����4�       ��2	���ǩ�A�*

epsilon�m��0��.       ��W�	d ��ǩ�A�* 

Average reward per step�m��ǎ��       ��2	!��ǩ�A�*

epsilon�m��`�Y.       ��W�	�F��ǩ�A�* 

Average reward per step�m��g�/5       ��2	�G��ǩ�A�*

epsilon�m��׃�.       ��W�	�b��ǩ�A�* 

Average reward per step�m���`�       ��2	Id��ǩ�A�*

epsilon�m���Nf.       ��W�	���ǩ�A�* 

Average reward per step�m��T\R�       ��2	е��ǩ�A�*

epsilon�m����U�.       ��W�	���ǩ�A�* 

Average reward per step�m�����       ��2	���ǩ�A�*

epsilon�m���9 %.       ��W�	χ��ǩ�A�* 

Average reward per step�m���C�       ��2	j���ǩ�A�*

epsilon�m���˓�.       ��W�	�{��ǩ�A�* 

Average reward per step�m���֩�       ��2	-|��ǩ�A�*

epsilon�m���h)s0       ���_	R���ǩ�A=*#
!
Average reward per episodeUU��/jz.       ��W�	���ǩ�A=*!

total reward per episode  |��ef�.       ��W�	N���ǩ�A�* 

Average reward per stepUU�����       ��2	���ǩ�A�*

epsilonUU��K�.       ��W�	`���ǩ�A�* 

Average reward per stepUU���!�       ��2	`���ǩ�A�*

epsilonUU���.       ��W�	r��ǩ�A�* 

Average reward per stepUU�����q       ��2	��ǩ�A�*

epsilonUU��p��.       ��W�	�{��ǩ�A�* 

Average reward per stepUU����`8       ��2	_|��ǩ�A�*

epsilonUU��]��.       ��W�	+���ǩ�A�* 

Average reward per stepUU�����       ��2	ӣ��ǩ�A�*

epsilonUU��6 �.       ��W�	b���ǩ�A�* 

Average reward per stepUU����^�       ��2	{���ǩ�A�*

epsilonUU��d}.       ��W�	����ǩ�A�* 

Average reward per stepUU����G       ��2	����ǩ�A�*

epsilonUU���}�.       ��W�	�k��ǩ�A�* 

Average reward per stepUU����nj       ��2	�l��ǩ�A�*

epsilonUU��n��.       ��W�	z���ǩ�A�* 

Average reward per stepUU���P�V       ��2	O���ǩ�A�*

epsilonUU����j.       ��W�	W���ǩ�A�* 

Average reward per stepUU���֯       ��2	!���ǩ�A�*

epsilonUU��%R`�.       ��W�	���ǩ�A�* 

Average reward per stepUU����V�       ��2	t��ǩ�A�*

epsilonUU�����.       ��W�	%#��ǩ�A�* 

Average reward per stepUU���
�       ��2	[$��ǩ�A�*

epsilonUU�����k.       ��W�	���ǩ�A�* 

Average reward per stepUU���|v:       ��2	���ǩ�A�*

epsilonUU���8.       ��W�	��ǩ�A�* 

Average reward per stepUU���bT       ��2	��ǩ�A�*

epsilonUU������.       ��W�	~��ǩ�A�* 

Average reward per stepUU���Q��       ��2	���ǩ�A�*

epsilonUU��;,.       ��W�	>y�ǩ�A�* 

Average reward per stepUU�����/       ��2	�{�ǩ�A�*

epsilonUU�����.       ��W�	&��ǩ�A�* 

Average reward per stepUU��Ǿ2       ��2	r��ǩ�A�*

epsilonUU��h��.       ��W�	
�ǩ�A�* 

Average reward per stepUU��ƿ��       ��2	��ǩ�A�*

epsilonUU��%~v�.       ��W�	���ǩ�A�* 

Average reward per stepUU�����       ��2	���ǩ�A�*

epsilonUU��*fa�.       ��W�	���ǩ�A�* 

Average reward per stepUU����*�       ��2	_��ǩ�A�*

epsilonUU�����I.       ��W�		�!�ǩ�A�* 

Average reward per stepUU���|��       ��2	 "�ǩ�A�*

epsilonUU��ܻ��.       ��W�	 <$�ǩ�A�* 

Average reward per stepUU��M�3       ��2	�<$�ǩ�A�*

epsilonUU��#*��.       ��W�	e&�ǩ�A�* 

Average reward per stepUU���LB       ��2	�e&�ǩ�A�*

epsilonUU���o�u.       ��W�	�.(�ǩ�A�* 

Average reward per stepUU�����       ��2	8/(�ǩ�A�*

epsilonUU���-�w.       ��W�	A�*�ǩ�A�* 

Average reward per stepUU����C       ��2	E�*�ǩ�A�*

epsilonUU���Za.       ��W�	�.�ǩ�A�* 

Average reward per stepUU��/!F�       ��2	��.�ǩ�A�*

epsilonUU���.       ��W�	��1�ǩ�A�* 

Average reward per stepUU���J3F       ��2	��1�ǩ�A�*

epsilonUU����97.       ��W�	��4�ǩ�A�* 

Average reward per stepUU��'�T       ��2	v�4�ǩ�A�*

epsilonUU��z���.       ��W�	�6�ǩ�A�* 

Average reward per stepUU���]N�       ��2	��6�ǩ�A�*

epsilonUU��'MB�.       ��W�	W�8�ǩ�A�* 

Average reward per stepUU����.C       ��2	��8�ǩ�A�*

epsilonUU��޾[�.       ��W�	��:�ǩ�A�* 

Average reward per stepUU����C2       ��2	{�:�ǩ�A�*

epsilonUU��-�bK.       ��W�	+�<�ǩ�A�* 

Average reward per stepUU������       ��2	��<�ǩ�A�*

epsilonUU���XQ=.       ��W�	~r>�ǩ�A�* 

Average reward per stepUU��HZ�       ��2	!s>�ǩ�A�*

epsilonUU���#\�.       ��W�	��?�ǩ�A�* 

Average reward per stepUU������       ��2	��?�ǩ�A�*

epsilonUU���V�m.       ��W�	�B�ǩ�A�* 

Average reward per stepUU��v�E�       ��2	KB�ǩ�A�*

epsilonUU���%6�.       ��W�	�DD�ǩ�A�* 

Average reward per stepUU����v}       ��2	�ED�ǩ�A�*

epsilonUU��O���.       ��W�	��E�ǩ�A�* 

Average reward per stepUU��r3�       ��2	��E�ǩ�A�*

epsilonUU����j�.       ��W�	-"G�ǩ�A�* 

Average reward per stepUU���<�       ��2	�"G�ǩ�A�*

epsilonUU��@��.       ��W�	�/I�ǩ�A�* 

Average reward per stepUU��M���       ��2	Y0I�ǩ�A�*

epsilonUU���@&.       ��W�	
eK�ǩ�A�* 

Average reward per stepUU����6�       ��2	�eK�ǩ�A�*

epsilonUU��ȭ�j.       ��W�	��M�ǩ�A�* 

Average reward per stepUU��|��f       ��2	Q�M�ǩ�A�*

epsilonUU��+��h.       ��W�	A}O�ǩ�A�* 

Average reward per stepUU�����8       ��2	~O�ǩ�A�*

epsilonUU����.       ��W�	)&Q�ǩ�A�* 

Average reward per stepUU��M�]       ��2	�&Q�ǩ�A�*

epsilonUU��P.�0       ���_	hBQ�ǩ�A>*#
!
Average reward per episode0��9	qR.       ��W�	�BQ�ǩ�A>*!

total reward per episode  ��P���.       ��W�	�U�ǩ�A�* 

Average reward per step0��ɾ/v       ��2	��U�ǩ�A�*

epsilon0����E.       ��W�	B�W�ǩ�A�* 

Average reward per step0��/to$       ��2	[�W�ǩ�A�*

epsilon0���R��.       ��W�	xBY�ǩ�A�* 

Average reward per step0��jI��       ��2	CY�ǩ�A�*

epsilon0���:�.       ��W�	&T[�ǩ�A�* 

Average reward per step0��H��       ��2	�T[�ǩ�A�*

epsilon0������.       ��W�	�l]�ǩ�A�* 

Average reward per step0��C�s,       ��2	�m]�ǩ�A�*

epsilon0���y�r.       ��W�	�_�ǩ�A�* 

Average reward per step0��#ݾ       ��2	��_�ǩ�A�*

epsilon0��� �.       ��W�	�a�ǩ�A�* 

Average reward per step0��!���       ��2	��a�ǩ�A�*

epsilon0���Fl�.       ��W�	��c�ǩ�A�* 

Average reward per step0��� X�       ��2	m�c�ǩ�A�*

epsilon0���&�~.       ��W�	5f�ǩ�A�* 

Average reward per step0��3�s$       ��2	�f�ǩ�A�*

epsilon0��g)�O.       ��W�	�h�ǩ�A�* 

Average reward per step0��FjA       ��2	vh�ǩ�A�*

epsilon0�����.       ��W�	@/j�ǩ�A�* 

Average reward per step0��\���       ��2	0j�ǩ�A�*

epsilon0��f.�.       ��W�	�jl�ǩ�A�* 

Average reward per step0���\h       ��2	�kl�ǩ�A�*

epsilon0��V)�k.       ��W�	r�m�ǩ�A�* 

Average reward per step0��o�8�       ��2	?�m�ǩ�A�*

epsilon0��£\.       ��W�	 %p�ǩ�A�* 

Average reward per step0����       ��2	�%p�ǩ�A�*

epsilon0���S�0.       ��W�	�Ir�ǩ�A�* 

Average reward per step0�����       ��2	�Jr�ǩ�A�*

epsilon0��Ct@�.       ��W�	Act�ǩ�A�* 

Average reward per step0��r�΍       ��2	4et�ǩ�A�*

epsilon0��e�.       ��W�	��v�ǩ�A�* 

Average reward per step0��}�o       ��2	��v�ǩ�A�*

epsilon0��B��W.       ��W�	&Sx�ǩ�A�* 

Average reward per step0��L�}       ��2	eTx�ǩ�A�*

epsilon0���p�.       ��W�	3�z�ǩ�A�* 

Average reward per step0���3��       ��2	�z�ǩ�A�*

epsilon0���p~.       ��W�	P�|�ǩ�A�* 

Average reward per step0��r.��       ��2	�|�ǩ�A�*

epsilon0��Dr��.       ��W�	>�~�ǩ�A�* 

Average reward per step0�����       ��2	��~�ǩ�A�*

epsilon0����.       ��W�	.���ǩ�A�* 

Average reward per step0�����       ��2	e���ǩ�A�*

epsilon0��}�-9.       ��W�	`���ǩ�A�* 

Average reward per step0���B5	       ��2	%���ǩ�A�*

epsilon0�����I.       ��W�	�I��ǩ�A�* 

Average reward per step0��p��       ��2	bJ��ǩ�A�*

epsilon0������.       ��W�	.��ǩ�A�* 

Average reward per step0��#��R       ��2	���ǩ�A�*

epsilon0��w��.       ��W�	���ǩ�A�* 

Average reward per step0�����       ��2	%��ǩ�A�*

epsilon0��Z�ן.       ��W�	7��ǩ�A�* 

Average reward per step0���<�       ��2	���ǩ�A�*

epsilon0���U`�.       ��W�	�4��ǩ�A�* 

Average reward per step0����&       ��2	�5��ǩ�A�*

epsilon0����.       ��W�	�D��ǩ�A�* 

Average reward per step0����Q       ��2	�E��ǩ�A�*

epsilon0��§��.       ��W�	쇖�ǩ�A�* 

Average reward per step0���U�D       ��2	���ǩ�A�*

epsilon0��A�.       ��W�	���ǩ�A�* 

Average reward per step0���       ��2	����ǩ�A�*

epsilon0��?4��.       ��W�	
L��ǩ�A�* 

Average reward per step0��dA+       ��2	�L��ǩ�A�*

epsilon0��5*�y.       ��W�	�x��ǩ�A�* 

Average reward per step0���l�x       ��2	�y��ǩ�A�*

epsilon0����$6.       ��W�	_���ǩ�A�* 

Average reward per step0��	�Y       ��2	����ǩ�A�*

epsilon0��@���.       ��W�	4���ǩ�A�* 

Average reward per step0��Ғ�       ��2	����ǩ�A�*

epsilon0���}�.       ��W�	cb��ǩ�A�* 

Average reward per step0�����       ��2	c��ǩ�A�*

epsilon0��/��H.       ��W�	�I��ǩ�A�* 

Average reward per step0��Z�       ��2	AJ��ǩ�A�*

epsilon0���wW..       ��W�	[��ǩ�A�* 

Average reward per step0������       ��2	S\��ǩ�A�*

epsilon0��,pF.       ��W�	XX��ǩ�A�* 

Average reward per step0���@       ��2	OY��ǩ�A�*

epsilon0���~/�.       ��W�	,֯�ǩ�A�* 

Average reward per step0���y��       ��2	�֯�ǩ�A�*

epsilon0��;nڝ.       ��W�	����ǩ�A�* 

Average reward per step0�����]       ��2	��ǩ�A�*

epsilon0�����C.       ��W�	�F��ǩ�A�* 

Average reward per step0��uV��       ��2	H��ǩ�A�*

epsilon0���{��.       ��W�	�\��ǩ�A�* 

Average reward per step0���-)       ��2	�]��ǩ�A�*

epsilon0��h���.       ��W�	���ǩ�A�* 

Average reward per step0����iS       ��2	`��ǩ�A�*

epsilon0���U��.       ��W�	�)��ǩ�A�* 

Average reward per step0��\@ai       ��2	s*��ǩ�A�*

epsilon0�����.       ��W�	�\��ǩ�A�* 

Average reward per step0��2iP�       ��2	y]��ǩ�A�*

epsilon0��FiU.       ��W�	����ǩ�A�* 

Average reward per step0��i\u$       ��2	|���ǩ�A�*

epsilon0��~��.       ��W�	�-��ǩ�A�* 

Average reward per step0��VEw�       ��2	8.��ǩ�A�*

epsilon0��@c}�.       ��W�	����ǩ�A�* 

Average reward per step0����       ��2	J���ǩ�A�*

epsilon0��<"O0       ���_	E���ǩ�A?*#
!
Average reward per episode����Rl.       ��W�	����ǩ�A?*!

total reward per episode  ��C��.       ��W�	�B��ǩ�A�* 

Average reward per step���Jy�       ��2	cC��ǩ�A�*

epsilon��^?.       ��W�	sc��ǩ�A�* 

Average reward per step��(\��       ��2	d��ǩ�A�*

epsilon���C��.       ��W�	ޮ��ǩ�A�* 

Average reward per step����z       ��2	}���ǩ�A�*

epsilon��E�Ă.       ��W�	��ǩ�A�* 

Average reward per step��ј;       ��2	���ǩ�A�*

epsilon��5Z�e.       ��W�	�w��ǩ�A�* 

Average reward per step����a       ��2	dx��ǩ�A�*

epsilon��՛%g.       ��W�	����ǩ�A�* 

Average reward per step��j�       ��2	����ǩ�A�*

epsilon��!��.       ��W�	����ǩ�A�* 

Average reward per step��<"       ��2	����ǩ�A�*

epsilon���1\&.       ��W�	W���ǩ�A�* 

Average reward per step��#���       ��2	���ǩ�A�*

epsilon���9Ƴ.       ��W�	{��ǩ�A�* 

Average reward per step�����Y       ��2	H��ǩ�A�*

epsilon�����.       ��W�	7���ǩ�A�* 

Average reward per step�����       ��2	���ǩ�A�*

epsilon��G{�n.       ��W�	����ǩ�A�* 

Average reward per step���V�       ��2	s���ǩ�A�*

epsilon����s.       ��W�	����ǩ�A�* 

Average reward per step���wg       ��2	����ǩ�A�*

epsilon���<��.       ��W�	�*��ǩ�A�* 

Average reward per step���,�1       ��2	N+��ǩ�A�*

epsilon���\i0       ���_	E��ǩ�A@*#
!
Average reward per episode��I��P.       ��W�	�E��ǩ�A@*!

total reward per episode  $��	�.       ��W�	����ǩ�A�* 

Average reward per step��I��si�       ��2	����ǩ�A�*

epsilon��I�kq�.       ��W�	���ǩ�A�* 

Average reward per step��I�q�z       ��2	���ǩ�A�*

epsilon��I�d��.       ��W�	L��ǩ�A�* 

Average reward per step��I�@��       ��2	���ǩ�A�*

epsilon��I�<��Y.       ��W�	�+��ǩ�A�* 

Average reward per step��I��tN$       ��2	g,��ǩ�A�*

epsilon��I���.       ��W�	�E��ǩ�A�* 

Average reward per step��I�8��1       ��2	tF��ǩ�A�*

epsilon��I�O��z.       ��W�	�_��ǩ�A�* 

Average reward per step��I� ��       ��2	�`��ǩ�A�*

epsilon��I���.       ��W�	w���ǩ�A�* 

Average reward per step��I���'       ��2	Q���ǩ�A�*

epsilon��I��P~�.       ��W�	���ǩ�A�* 

Average reward per step��I��!�       ��2	����ǩ�A�*

epsilon��I���^.       ��W�	����ǩ�A�* 

Average reward per step��I�C���       ��2	f���ǩ�A�*

epsilon��I����E.       ��W�	�� �ǩ�A�* 

Average reward per step��I�^��       ��2	�� �ǩ�A�*

epsilon��I��{x�.       ��W�	�f�ǩ�A�* 

Average reward per step��I��7~�       ��2	{g�ǩ�A�*

epsilon��I���~=.       ��W�	���ǩ�A�* 

Average reward per step��I�}sy�       ��2	���ǩ�A�*

epsilon��I�|D��.       ��W�	���ǩ�A�* 

Average reward per step��I���¥       ��2	���ǩ�A�*

epsilon��I���M.       ��W�	&	�ǩ�A�* 

Average reward per step��I��8��       ��2	�	�ǩ�A�*

epsilon��I�:|�>.       ��W�	u;�ǩ�A�* 

Average reward per step��I�Pa�.       ��2	<�ǩ�A�*

epsilon��I���t/.       ��W�	Q�ǩ�A�* 

Average reward per step��I��# �       ��2	�Q�ǩ�A�*

epsilon��I� ��.       ��W�	���ǩ�A�* 

Average reward per step��I�u��       ��2	w��ǩ�A�*

epsilon��I�چ�.       ��W�	O�ǩ�A�* 

Average reward per step��I�X�       ��2	�ǩ�A�*

epsilon��I�S�}.       ��W�	�z�ǩ�A�* 

Average reward per step��I��k<t       ��2	|{�ǩ�A�*

epsilon��I��ANI.       ��W�	��ǩ�A�* 

Average reward per step��I��6�{       ��2	D�ǩ�A�*

epsilon��I��X��.       ��W�	�Z�ǩ�A�* 

Average reward per step��I�޽�       ��2	B[�ǩ�A�*

epsilon��I���TJ.       ��W�	��ǩ�A�* 

Average reward per step��I��x�       ��2	���ǩ�A�*

epsilon��I�Ҁl�.       ��W�	���ǩ�A�* 

Average reward per step��I�®[r       ��2	L��ǩ�A�*

epsilon��I��$m.       ��W�	R&�ǩ�A�* 

Average reward per step��I�.�H       ��2	�&�ǩ�A�*

epsilon��I����D.       ��W�	�N�ǩ�A�* 

Average reward per step��I�j�D�       ��2	LO�ǩ�A�*

epsilon��I���v�.       ��W�	�!�ǩ�A�* 

Average reward per step��I��9Q@       ��2	��!�ǩ�A�*

epsilon��I��6.       ��W�	��#�ǩ�A�* 

Average reward per step��I�[�xH       ��2	f�#�ǩ�A�*

epsilon��I��L�N.       ��W�	��%�ǩ�A�* 

Average reward per step��I�Qտ�       ��2	_�%�ǩ�A�*

epsilon��I��T�/.       ��W�	m (�ǩ�A�* 

Average reward per step��I��ߛj       ��2	(�ǩ�A�*

epsilon��I�'�
@.       ��W�	#0*�ǩ�A�* 

Average reward per step��I��R��       ��2	�0*�ǩ�A�*

epsilon��I��"�J.       ��W�	�,�ǩ�A�* 

Average reward per step��I�W�z�       ��2	��,�ǩ�A�*

epsilon��I��9J0       ���_	i�,�ǩ�AA*#
!
Average reward per episode�1���.�.       ��W�	D�,�ǩ�AA*!

total reward per episode  ýj@~.       ��W�	��0�ǩ�A�* 

Average reward per step�1��~�nT       ��2	o�0�ǩ�A�*

epsilon�1���=�.       ��W�	��2�ǩ�A�* 

Average reward per step�1�� W5       ��2	�2�ǩ�A�*

epsilon�1�����.       ��W�	�+5�ǩ�A�* 

Average reward per step�1�����       ��2	�,5�ǩ�A�*

epsilon�1���^.       ��W�	�L7�ǩ�A�* 

Average reward per step�1��&�C       ��2	�M7�ǩ�A�*

epsilon�1����<�.       ��W�	�i9�ǩ�A�* 

Average reward per step�1��+�M       ��2	8k9�ǩ�A�*

epsilon�1�����.       ��W�	��:�ǩ�A�* 

Average reward per step�1����Ea       ��2	z ;�ǩ�A�*

epsilon�1����z=.       ��W�	�5=�ǩ�A�* 

Average reward per step�1��y@��       ��2	L6=�ǩ�A�*

epsilon�1���~9.       ��W�	�K?�ǩ�A�* 

Average reward per step�1���*��       ��2	nL?�ǩ�A�*

epsilon�1���6�.       ��W�	_�A�ǩ�A�* 

Average reward per step�1��z&'       ��2	9�A�ǩ�A�*

epsilon�1��B{��.       ��W�	�C�ǩ�A�* 

Average reward per step�1���F�l       ��2	�C�ǩ�A�*

epsilon�1���G�.       ��W�	�?E�ǩ�A�* 

Average reward per step�1��^�V�       ��2	�@E�ǩ�A�*

epsilon�1��#.       ��W�	mG�ǩ�A�* 

Average reward per step�1���3       ��2	�mG�ǩ�A�*

epsilon�1�����~.       ��W�	T�I�ǩ�A�* 

Average reward per step�1��w�Ap       ��2	"�I�ǩ�A�*

epsilon�1��?��.       ��W�	�K�ǩ�A�* 

Average reward per step�1�����       ��2	��K�ǩ�A�*

epsilon�1��`�~b.       ��W�	.N�ǩ�A�* 

Average reward per step�1��z���       ��2	�N�ǩ�A�*

epsilon�1��%��H.       ��W�	�O�ǩ�A�* 

Average reward per step�1���^p       ��2	|�O�ǩ�A�*

epsilon�1��}#/.       ��W�	ʍQ�ǩ�A�* 

Average reward per step�1��a�       ��2	e�Q�ǩ�A�*

epsilon�1��璔�.       ��W�	�T�ǩ�A�* 

Average reward per step�1��V�       ��2	��T�ǩ�A�*

epsilon�1����0       ���_		�T�ǩ�AB*#
!
Average reward per episodeUU�Y���.       ��W�	��T�ǩ�AB*!

total reward per episode  ���<.       ��W�	o�Y�ǩ�A�* 

Average reward per stepUU�^,       ��2	<�Y�ǩ�A�*

epsilonUU���^.       ��W�	�[�ǩ�A�* 

Average reward per stepUU��#�%       ��2	��[�ǩ�A�*

epsilonUU����}.       ��W�	��]�ǩ�A�* 

Average reward per stepUU����       ��2	s�]�ǩ�A�*

epsilonUU�s�9^.       ��W�	K�_�ǩ�A�* 

Average reward per stepUU��x]g       ��2	��_�ǩ�A�*

epsilonUU�YSp�.       ��W�	��b�ǩ�A�* 

Average reward per stepUU�TY��       ��2	��b�ǩ�A�*

epsilonUU����.       ��W�	vd�ǩ�A�* 

Average reward per stepUU�E��       ��2	;d�ǩ�A�*

epsilonUU���.       ��W�	�3f�ǩ�A�* 

Average reward per stepUU�n��       ��2	�4f�ǩ�A�*

epsilonUU��tf�.       ��W�	MKh�ǩ�A�* 

Average reward per stepUU�;�C�       ��2	�Kh�ǩ�A�*

epsilonUU��l�.       ��W�	SXj�ǩ�A�* 

Average reward per stepUU��bX�       ��2	�Xj�ǩ�A�*

epsilonUU�xi��.       ��W�	�l�ǩ�A�* 

Average reward per stepUU��sg       ��2	��l�ǩ�A�*

epsilonUU���"�.       ��W�	=�n�ǩ�A�* 

Average reward per stepUU��x�W       ��2	��n�ǩ�A�*

epsilonUU�'���.       ��W�	}�p�ǩ�A�* 

Average reward per stepUU��Q{~       ��2	!�p�ǩ�A�*

epsilonUU��(Sq.       ��W�		s�ǩ�A�* 

Average reward per stepUU����       ��2	�	s�ǩ�A�*

epsilonUU�e=�.       ��W�	_)u�ǩ�A�* 

Average reward per stepUU����       ��2	,*u�ǩ�A�*

epsilonUU���.       ��W�	 �v�ǩ�A�* 

Average reward per stepUU��)Gg       ��2	��v�ǩ�A�*

epsilonUU��]�?.       ��W�	;Ry�ǩ�A�* 

Average reward per stepUU�����       ��2	�Ry�ǩ�A�*

epsilonUU�dr�].       ��W�	
�{�ǩ�A�* 

Average reward per stepUU�c�       ��2	��{�ǩ�A�*

epsilonUU��K�.       ��W�	|�|�ǩ�A�* 

Average reward per stepUU�_       ��2	�|�ǩ�A�*

epsilonUU�I�^.       ��W�	��ǩ�A�* 

Average reward per stepUU��z:0       ��2	܀�ǩ�A�*

epsilonUU���+1.       ��W�	����ǩ�A�* 

Average reward per stepUU���X�       ��2	^��ǩ�A�*

epsilonUU�2"�.       ��W�	���ǩ�A�* 

Average reward per stepUU�Ⱦ�^       ��2	=��ǩ�A�*

epsilonUU�4�_.       ��W�	�D��ǩ�A�* 

Average reward per stepUU�ol�a       ��2	ZE��ǩ�A�*

epsilonUU�Z&).       ��W�	f��ǩ�A�* 

Average reward per stepUU�K7<       ��2	�f��ǩ�A�*

epsilonUU��	E�.       ��W�	o���ǩ�A�* 

Average reward per stepUU��C�S       ��2	@���ǩ�A�*

epsilonUU�"�^�.       ��W�	9'��ǩ�A�* 

Average reward per stepUU�$V�"       ��2	�'��ǩ�A�*

epsilonUU�9c�.       ��W�	C��ǩ�A�* 

Average reward per stepUU��SY�       ��2	�C��ǩ�A�*

epsilonUU����B.       ��W�	KW��ǩ�A�* 

Average reward per stepUU��nx       ��2	6X��ǩ�A�*

epsilonUU�J"(6.       ��W�	Uh��ǩ�A�* 

Average reward per stepUU�rז�       ��2	�h��ǩ�A�*

epsilonUU�ʥ�S.       ��W�	�w��ǩ�A�* 

Average reward per stepUU�W�        ��2	)x��ǩ�A�*

epsilonUU�z@g.       ��W�	Ͼ��ǩ�A�* 

Average reward per stepUU�Q*�       ��2	����ǩ�A�*

epsilonUU�{\L�.       ��W�	7��ǩ�A�* 

Average reward per stepUU��#�n       ��2	���ǩ�A�*

epsilonUU�;pk�.       ��W�	���ǩ�A�* 

Average reward per stepUU�?��       ��2	���ǩ�A�*

epsilonUU�?L�.       ��W�	u��ǩ�A�* 

Average reward per stepUU�e�Y�       ��2	%��ǩ�A�*

epsilonUU�?�.       ��W�	b���ǩ�A�* 

Average reward per stepUU�g�       ��2	���ǩ�A�*

epsilonUU���!_.       ��W�	����ǩ�A�* 

Average reward per stepUU�����       ��2	Ǽ��ǩ�A�*

epsilonUU�X�R.       ��W�	���ǩ�A�* 

Average reward per stepUU�,��m       ��2	���ǩ�A�*

epsilonUU�}R�.       ��W�	#N��ǩ�A�* 

Average reward per stepUU�OG�T       ��2	�N��ǩ�A�*

epsilonUU�Q���.       ��W�	�k��ǩ�A�* 

Average reward per stepUU���e�       ��2	�l��ǩ�A�*

epsilonUU���f�.       ��W�	����ǩ�A�* 

Average reward per stepUU�Gڠ       ��2	����ǩ�A�*

epsilonUU����.       ��W�	I��ǩ�A�* 

Average reward per stepUU�� �       ��2	���ǩ�A�*

epsilonUU�Q
�.       ��W�	�A��ǩ�A�* 

Average reward per stepUU�;�(�       ��2	�B��ǩ�A�*

epsilonUU�����.       ��W�	����ǩ�A�* 

Average reward per stepUU���m       ��2	Q���ǩ�A�*

epsilonUU�Q_+b.       ��W�	����ǩ�A�* 

Average reward per stepUU�����       ��2	����ǩ�A�*

epsilonUU�`?.       ��W�	~o��ǩ�A�* 

Average reward per stepUU�|e�2       ��2	Lp��ǩ�A�*

epsilonUU�rj��.       ��W�	^-��ǩ�A�* 

Average reward per stepUU�%9�       ��2	,.��ǩ�A�*

epsilonUU�H���.       ��W�	3���ǩ�A�* 

Average reward per stepUU����       ��2	����ǩ�A�*

epsilonUU�5�M�.       ��W�	����ǩ�A�* 

Average reward per stepUU�S��       ��2	b���ǩ�A�*

epsilonUU���4.       ��W�	$��ǩ�A�* 

Average reward per stepUU��	}d       ��2	���ǩ�A�*

epsilonUU��raS.       ��W�	֍��ǩ�A�* 

Average reward per stepUU���*�       ��2	����ǩ�A�*

epsilonUU�	�b�.       ��W�	D���ǩ�A�* 

Average reward per stepUU�c�F       ��2	���ǩ�A�*

epsilonUU��Rĩ.       ��W�	9���ǩ�A�* 

Average reward per stepUU�x�}e       ��2	���ǩ�A�*

epsilonUU��ç.       ��W�	 ���ǩ�A�* 

Average reward per stepUU�}��U       ��2	����ǩ�A�*

epsilonUU�%��V.       ��W�	X��ǩ�A�* 

Average reward per stepUU����i       ��2	���ǩ�A�*

epsilonUU��Xa.       ��W�	2���ǩ�A�* 

Average reward per stepUU�:Eʒ       ��2	���ǩ�A�*

epsilonUU�9�pz.       ��W�	m6��ǩ�A�* 

Average reward per stepUU�g�i�       ��2	7��ǩ�A�*

epsilonUU���:f.       ��W�	IH��ǩ�A�* 

Average reward per stepUU���       ��2	�H��ǩ�A�*

epsilonUU��1�n.       ��W�	����ǩ�A�* 

Average reward per stepUU��ӷ       ��2	O���ǩ�A�*

epsilonUU���a�.       ��W�	�#��ǩ�A�* 

Average reward per stepUU�N��       ��2	p$��ǩ�A�*

epsilonUU�4i�~.       ��W�	kF��ǩ�A�* 

Average reward per stepUU��v��       ��2	G��ǩ�A�*

epsilonUU�S_��.       ��W�	fi��ǩ�A�* 

Average reward per stepUU�\3WR       ��2	�i��ǩ�A�*

epsilonUU�x���.       ��W�	n��ǩ�A�* 

Average reward per stepUU�I.@�       ��2	��ǩ�A�*

epsilonUU��4"�.       ��W�	���ǩ�A�* 

Average reward per stepUU����w       ��2	���ǩ�A�*

epsilonUU�a=�	.       ��W�	���ǩ�A�* 

Average reward per stepUU���6�       ��2	v��ǩ�A�*

epsilonUU����{.       ��W�	���ǩ�A�* 

Average reward per stepUU�rM(�       ��2	���ǩ�A�*

epsilonUU�߳�.       ��W�	sH!�ǩ�A�* 

Average reward per stepUU��l�       ��2	I!�ǩ�A�*

epsilonUU�r�[�.       ��W�	�^#�ǩ�A�* 

Average reward per stepUU��n�       ��2	|_#�ǩ�A�*

epsilonUU�=:�k.       ��W�	�%�ǩ�A�* 

Average reward per stepUU��`#e       ��2	��%�ǩ�A�*

epsilonUU���G;.       ��W�	��'�ǩ�A�* 

Average reward per stepUU��K�Y       ��2	6�'�ǩ�A�*

epsilonUU�.R�6.       ��W�	��)�ǩ�A�* 

Average reward per stepUU�CCD       ��2	��)�ǩ�A�*

epsilonUU���x.       ��W�	,�ǩ�A�* 

Average reward per stepUU����       ��2	�,�ǩ�A�*

epsilonUU�"#e�.       ��W�	%�-�ǩ�A�* 

Average reward per stepUU��]�"       ��2	�-�ǩ�A�*

epsilonUU�ֻ�.       ��W�	�/�ǩ�A�* 

Average reward per stepUU��s�       ��2	�/�ǩ�A�*

epsilonUU��|ؕ.       ��W�	{1�ǩ�A�* 

Average reward per stepUU�,k�       ��2	01�ǩ�A�*

epsilonUU��6�Y.       ��W�	/03�ǩ�A�* 

Average reward per stepUU�c��       ��2	�03�ǩ�A�*

epsilonUU�{��e.       ��W�	�J5�ǩ�A�* 

Average reward per stepUU��L;       ��2	�K5�ǩ�A�*

epsilonUU�~.       ��W�	x�7�ǩ�A�* 

Average reward per stepUU�T���       ��2	I�7�ǩ�A�*

epsilonUU���.       ��W�	9&9�ǩ�A�* 

Average reward per stepUU���       ��2	�&9�ǩ�A�*

epsilonUU��1��.       ��W�	�/;�ǩ�A�* 

Average reward per stepUU�0�f       ��2	M0;�ǩ�A�*

epsilonUU�����.       ��W�	�C=�ǩ�A�* 

Average reward per stepUU���)L       ��2	_D=�ǩ�A�*

epsilonUU��@�k.       ��W�	�w?�ǩ�A�* 

Average reward per stepUU����       ��2	�x?�ǩ�A�*

epsilonUU����<.       ��W�	��A�ǩ�A�* 

Average reward per stepUU�m@�       ��2	m�A�ǩ�A�*

epsilonUU�9��.       ��W�	�WC�ǩ�A�* 

Average reward per stepUU�I~�;       ��2	qXC�ǩ�A�*

epsilonUU�(up�.       ��W�	�F�ǩ�A�* 

Average reward per stepUU�V3��       ��2	F�ǩ�A�*

epsilonUU�>��b.       ��W�	�J�ǩ�A�* 

Average reward per stepUU��$       ��2	�J�ǩ�A�*

epsilonUU���\�.       ��W�	O�K�ǩ�A�* 

Average reward per stepUU�T���       ��2	�K�ǩ�A�*

epsilonUU�CE|�.       ��W�	�O�ǩ�A�* 

Average reward per stepUU�$Ђ       ��2		�O�ǩ�A�*

epsilonUU�{�i�.       ��W�	�AR�ǩ�A�* 

Average reward per stepUU�)���       ��2	�BR�ǩ�A�*

epsilonUU�!\�.       ��W�	��S�ǩ�A�* 

Average reward per stepUU��A�       ��2	z�S�ǩ�A�*

epsilonUU�n���.       ��W�	��U�ǩ�A�* 

Average reward per stepUU�Ҳ �       ��2	��U�ǩ�A�*

epsilonUU��)�^.       ��W�	<X�ǩ�A�* 

Average reward per stepUU�Iۿ�       ��2	�<X�ǩ�A�*

epsilonUU�e+6z.       ��W�	��Y�ǩ�A�* 

Average reward per stepUU�~l�       ��2	��Y�ǩ�A�*

epsilonUU�Id`�.       ��W�	��[�ǩ�A�* 

Average reward per stepUU�����       ��2	o�[�ǩ�A�*

epsilonUU��XS�.       ��W�	�^�ǩ�A�* 

Average reward per stepUU���       ��2	9^�ǩ�A�*

epsilonUU� /E�.       ��W�	Re`�ǩ�A�* 

Average reward per stepUU���J^       ��2	4f`�ǩ�A�*

epsilonUU��7��.       ��W�	7�a�ǩ�A�* 

Average reward per stepUU���       ��2	��a�ǩ�A�*

epsilonUU��o�.       ��W�	T d�ǩ�A�* 

Average reward per stepUU�#��       ��2	d�ǩ�A�*

epsilonUU���:.       ��W�	tBf�ǩ�A�* 

Average reward per stepUU���p�       ��2	Cf�ǩ�A�*

epsilonUU�Sk"�.       ��W�	��h�ǩ�A�* 

Average reward per stepUU��;��       ��2	��h�ǩ�A�*

epsilonUU�W<	.       ��W�	I�j�ǩ�A�* 

Average reward per stepUU���Y<       ��2	0�j�ǩ�A�*

epsilonUU���L50       ���_	X�j�ǩ�AC*#
!
Average reward per episode�v�>�U�(.       ��W�	!�j�ǩ�AC*!

total reward per episode  �A�b��.       ��W�	�zp�ǩ�A�* 

Average reward per step�v�>9Ӛ�       ��2	�{p�ǩ�A�*

epsilon�v�>����.       ��W�	�r�ǩ�A�* 

Average reward per step�v�>��F       ��2	��r�ǩ�A�*

epsilon�v�>��/.       ��W�	��t�ǩ�A�* 

Average reward per step�v�>46�       ��2	o�t�ǩ�A�*

epsilon�v�>�)2E.       ��W�	�tv�ǩ�A�* 

Average reward per step�v�>g�       ��2	�uv�ǩ�A�*

epsilon�v�>�f^.       ��W�	��x�ǩ�A�* 

Average reward per step�v�>C-       ��2	y�x�ǩ�A�*

epsilon�v�>Q�0.       ��W�	��z�ǩ�A�* 

Average reward per step�v�>�?n       ��2	a�z�ǩ�A�*

epsilon�v�>u֣.       ��W�	
}�ǩ�A�* 

Average reward per step�v�>�.g       ��2	�
}�ǩ�A�*

epsilon�v�>��\.       ��W�	�#�ǩ�A�* 

Average reward per step�v�>�F��       ��2	5$�ǩ�A�*

epsilon�v�>��(.       ��W�	�S��ǩ�A�* 

Average reward per step�v�>BM       ��2	LT��ǩ�A�*

epsilon�v�>���.       ��W�	�q��ǩ�A�* 

Average reward per step�v�>�V�       ��2	�r��ǩ�A�*

epsilon�v�>g�[�.       ��W�	����ǩ�A�* 

Average reward per step�v�>��`)       ��2	����ǩ�A�*

epsilon�v�>�F�.       ��W�	l"��ǩ�A�* 

Average reward per step�v�>��+h       ��2	!#��ǩ�A�*

epsilon�v�>`��.       ��W�	�C��ǩ�A�* 

Average reward per step�v�>4��       ��2	D��ǩ�A�*

epsilon�v�>�k@�.       ��W�	{K��ǩ�A�* 

Average reward per step�v�>2�u       ��2	L��ǩ�A�*

epsilon�v�>���.       ��W�	k���ǩ�A�* 

Average reward per step�v�>igH       ��2	9���ǩ�A�*

epsilon�v�>�e�D.       ��W�	�C��ǩ�A�* 

Average reward per step�v�>g9,       ��2	�D��ǩ�A�*

epsilon�v�>e�c�.       ��W�	���ǩ�A�* 

Average reward per step�v�>��-I       ��2	Ѐ��ǩ�A�*

epsilon�v�>DԖ�.       ��W�	����ǩ�A�* 

Average reward per step�v�>ɞ�       ��2	ʥ��ǩ�A�*

epsilon�v�>.`ԋ.       ��W�	����ǩ�A�* 

Average reward per step�v�>�]�       ��2	����ǩ�A�*

epsilon�v�>�ڴz.       ��W�	����ǩ�A�* 

Average reward per step�v�>;ļ       ��2	#���ǩ�A�*

epsilon�v�>��(�.       ��W�	^՛�ǩ�A�* 

Average reward per step�v�>���       ��2	
֛�ǩ�A�*

epsilon�v�>���:.       ��W�	9��ǩ�A�* 

Average reward per step�v�>*�%       ��2	���ǩ�A�*

epsilon�v�>Z���.       ��W�	u"��ǩ�A�* 

Average reward per step�v�>m���       ��2	#��ǩ�A�*

epsilon�v�>	�?W.       ��W�	����ǩ�A�* 

Average reward per step�v�>���5       ��2	O���ǩ�A�*

epsilon�v�>���Z.       ��W�	����ǩ�A�* 

Average reward per step�v�> �o�       ��2	/���ǩ�A�*

epsilon�v�>��*�.       ��W�	
,��ǩ�A�* 

Average reward per step�v�>�GQ        ��2	�,��ǩ�A�*

epsilon�v�>潻.       ��W�	J��ǩ�A�* 

Average reward per step�v�>�T��       ��2	�J��ǩ�A�*

epsilon�v�>ҘY�.       ��W�	�k��ǩ�A�* 

Average reward per step�v�>97�       ��2	rl��ǩ�A�*

epsilon�v�>y�0       ���_	ӆ��ǩ�AD*#
!
Average reward per episodenۮ��I�w.       ��W�	]���ǩ�AD*!

total reward per episode  �����.       ��W�	�F��ǩ�A�* 

Average reward per stepnۮ��S�       ��2	(G��ǩ�A�*

epsilonnۮ��9\L.       ��W�	;r��ǩ�A�* 

Average reward per stepnۮ�&`�d       ��2	�r��ǩ�A�*

epsilonnۮ�W�>.       ��W�	$���ǩ�A�* 

Average reward per stepnۮ�� >       ��2	��ǩ�A�*

epsilonnۮ�Pv�`.       ��W�	�^��ǩ�A�* 

Average reward per stepnۮ���6G       ��2	[_��ǩ�A�*

epsilonnۮ��X��.       ��W�	g��ǩ�A�* 

Average reward per stepnۮ�uaj       ��2	�g��ǩ�A�*

epsilonnۮ�=�W<.       ��W�	�n��ǩ�A�* 

Average reward per stepnۮ�O> ]       ��2	"o��ǩ�A�*

epsilonnۮ��	�*.       ��W�	}��ǩ�A�* 

Average reward per stepnۮ�k���       ��2	J~��ǩ�A�*

epsilonnۮ��dK%.       ��W�	%���ǩ�A�* 

Average reward per stepnۮ�[Ӊ       ��2	ѭ��ǩ�A�*

epsilonnۮ��7T�.       ��W�	)ϼ�ǩ�A�* 

Average reward per stepnۮ�z`�       ��2	�ϼ�ǩ�A�*

epsilonnۮ�#�U.       ��W�	��ǩ�A�* 

Average reward per stepnۮ�kHSA       ��2	���ǩ�A�*

epsilonnۮ�,��.       ��W�	x���ǩ�A�* 

Average reward per stepnۮ���n�       ��2	-���ǩ�A�*

epsilonnۮ��-Y�.       ��W�	�&��ǩ�A�* 

Average reward per stepnۮ�H���       ��2	1'��ǩ�A�*

epsilonnۮ�;o��.       ��W�	K9��ǩ�A�* 

Average reward per stepnۮ��vf}       ��2	:��ǩ�A�*

epsilonnۮ��K.       ��W�	�\��ǩ�A�* 

Average reward per stepnۮ�|�@       ��2	]��ǩ�A�*

epsilonnۮ���^.       ��W�	<���ǩ�A�* 

Average reward per stepnۮ�ҳ\�       ��2	���ǩ�A�*

epsilonnۮ���C .       ��W�	����ǩ�A�* 

Average reward per stepnۮ�%� 7       ��2	6���ǩ�A�*

epsilonnۮ�����.       ��W�	���ǩ�A�* 

Average reward per stepnۮ�,~"       ��2	� ��ǩ�A�*

epsilonnۮ�JPc.       ��W�	zR��ǩ�A�* 

Average reward per stepnۮ���|       ��2	]S��ǩ�A�*

epsilonnۮ���y�.       ��W�	�|��ǩ�A�* 

Average reward per stepnۮ��k��       ��2	g}��ǩ�A�*

epsilonnۮ��%�b.       ��W�	g���ǩ�A�* 

Average reward per stepnۮ����       ��2	���ǩ�A�*

epsilonnۮ��z�.       ��W�	����ǩ�A�* 

Average reward per stepnۮ��wY       ��2	C ��ǩ�A�*

epsilonnۮ���`.       ��W�	�<��ǩ�A�* 

Average reward per stepnۮ���r       ��2	�=��ǩ�A�*

epsilonnۮ�~��2.       ��W�	a��ǩ�A�* 

Average reward per stepnۮ���       ��2	�a��ǩ�A�*

epsilonnۮ���н.       ��W�	H���ǩ�A�* 

Average reward per stepnۮ�c/�K       ��2	���ǩ�A�*

epsilonnۮ���.       ��W�	����ǩ�A�* 

Average reward per stepnۮ�%F*�       ��2	\���ǩ�A�*

epsilonnۮ�O���.       ��W�	�W��ǩ�A�* 

Average reward per stepnۮ��>��       ��2	�X��ǩ�A�*

epsilonnۮ����0       ���_	ut��ǩ�AE*#
!
Average reward per episode���3��C.       ��W�	�t��ǩ�AE*!

total reward per episode  �6�S�.       ��W�	����ǩ�A�* 

Average reward per step����h%a       ��2	����ǩ�A�*

epsilon���4��F.       ��W�	���ǩ�A�* 

Average reward per step���0�       ��2	���ǩ�A�*

epsilon���1	�.       ��W�	z��ǩ�A�* 

Average reward per step���C;�       ��2	��ǩ�A�*

epsilon����䧨.       ��W�	�F��ǩ�A�* 

Average reward per step����E       ��2	�G��ǩ�A�*

epsilon���ǙY.       ��W�	̵��ǩ�A�* 

Average reward per step�����!�       ��2	����ǩ�A�*

epsilon���,��.       ��W�	i6��ǩ�A�* 

Average reward per step���mF+c       ��2	;7��ǩ�A�*

epsilon���G�~.       ��W�	�d��ǩ�A�* 

Average reward per step����g��       ��2	�e��ǩ�A�*

epsilon�����j.       ��W�	����ǩ�A�* 

Average reward per step�����2       ��2	o���ǩ�A�*

epsilon���8̱#.       ��W�	_���ǩ�A�* 

Average reward per step���ʖJ�       ��2	$���ǩ�A�*

epsilon������ .       ��W�	$��ǩ�A�* 

Average reward per step������       ��2	�$��ǩ�A�*

epsilon�����.       ��W�	1��ǩ�A�* 

Average reward per step���Lx�5       ��2	2��ǩ�A�*

epsilon����pQ�.       ��W�	�]��ǩ�A�* 

Average reward per step���#�O~       ��2	�^��ǩ�A�*

epsilon���� �L.       ��W�	ʋ��ǩ�A�* 

Average reward per step����Vr�       ��2	����ǩ�A�*

epsilon���N`K.       ��W�	9�ǩ�A�* 

Average reward per step���V��]       ��2	��ǩ�A�*

epsilon���ܯ�.       ��W�	�E�ǩ�A�* 

Average reward per step����{�8       ��2	ͱE�ǩ�A�*

epsilon���1�~�.       ��W�	��G�ǩ�A�* 

Average reward per step���>8�       ��2	��G�ǩ�A�*

epsilon�����ET.       ��W�	" J�ǩ�A�* 

Average reward per step���1?J,       ��2	J�ǩ�A�*

epsilon������a.       ��W�	�K�ǩ�A�* 

Average reward per step������       ��2	c�K�ǩ�A�*

epsilon����JUi.       ��W�	��M�ǩ�A�* 

Average reward per step���6!�P       ��2	��M�ǩ�A�*

epsilon������X.       ��W�	��O�ǩ�A�* 

Average reward per step�����g�       ��2	��O�ǩ�A�*

epsilon���Q#�G.       ��W�	}$R�ǩ�A�* 

Average reward per step���f �N       ��2	S%R�ǩ�A�*

epsilon���X��.       ��W�	O�S�ǩ�A�* 

Average reward per step���8�f       ��2	6�S�ǩ�A�*

epsilon����Jvg.       ��W�	_V�ǩ�A�* 

Average reward per step����*�       ��2	WV�ǩ�A�*

epsilon���&�[�.       ��W�	TRX�ǩ�A�* 

Average reward per step����Q�J       ��2	aSX�ǩ�A�*

epsilon���읺�.       ��W�	P�Y�ǩ�A�* 

Average reward per step����R3�       ��2	:�Y�ǩ�A�*

epsilon���J�(.       ��W�	\�ǩ�A�* 

Average reward per step�����.       ��2	/\�ǩ�A�*

epsilon����@��0       ���_	�U\�ǩ�AF*#
!
Average reward per episode�ľ�;��.       ��W�	�V\�ǩ�AF*!

total reward per episode  �M��.       ��W�	f`�ǩ�A�* 

Average reward per step�ľ�$=$       ��2	`�ǩ�A�*

epsilon�ľ���!�.       ��W�	y:b�ǩ�A�* 

Average reward per step�ľ�ey��       ��2	�;b�ǩ�A�*

epsilon�ľ���p~.       ��W�	�d�ǩ�A�* 

Average reward per step�ľ��o�l       ��2	�d�ǩ�A�*

epsilon�ľ�&e�.       ��W�	��f�ǩ�A�* 

Average reward per step�ľ�ٝo�       ��2	X�f�ǩ�A�*

epsilon�ľ�1�.       ��W�	��h�ǩ�A�* 

Average reward per step�ľ����'       ��2	��h�ǩ�A�*

epsilon�ľ��n?�.       ��W�	.Wj�ǩ�A�* 

Average reward per step�ľ�2��       ��2	�Wj�ǩ�A�*

epsilon�ľ�����.       ��W�	�}l�ǩ�A�* 

Average reward per step�ľ�Y��       ��2	�~l�ǩ�A�*

epsilon�ľ��G;c.       ��W�	d�n�ǩ�A�* 

Average reward per step�ľ��h�       ��2	�n�ǩ�A�*

epsilon�ľ���78.       ��W�	��p�ǩ�A�* 

Average reward per step�ľ�[¥       ��2	��p�ǩ�A�*

epsilon�ľ�}?�{.       ��W�	*s�ǩ�A�* 

Average reward per step�ľ�qJ       ��2	�s�ǩ�A�*

epsilon�ľ�JfL�.       ��W�	^u�ǩ�A�* 

Average reward per step�ľ�ЭhI       ��2	g_u�ǩ�A�*

epsilon�ľ�7Ӑ�.       ��W�	�w�ǩ�A�* 

Average reward per step�ľ�'�m�       ��2	2w�ǩ�A�*

epsilon�ľ��z�.       ��W�	g&y�ǩ�A�* 

Average reward per step�ľ���K@       ��2	�&y�ǩ�A�*

epsilon�ľ���2�.       ��W�	Q{�ǩ�A�* 

Average reward per step�ľ�Bڨ~       ��2	�Q{�ǩ�A�*

epsilon�ľ�u/�.       ��W�	�u}�ǩ�A�* 

Average reward per step�ľ��HP       ��2	�v}�ǩ�A�*

epsilon�ľ����.       ��W�	���ǩ�A�* 

Average reward per step�ľ�N��(       ��2	��ǩ�A�*

epsilon�ľ�yc��.       ��W�	w��ǩ�A�* 

Average reward per step�ľ���e�       ��2	f��ǩ�A�*

epsilon�ľ�9�7�.       ��W�	k���ǩ�A�* 

Average reward per step�ľ��w�|       ��2	0���ǩ�A�*

epsilon�ľ��{.       ��W�	�݅�ǩ�A�* 

Average reward per step�ľ�h��       ��2	�ޅ�ǩ�A�*

epsilon�ľ��X��.       ��W�	�h��ǩ�A�* 

Average reward per step�ľ�����       ��2	�i��ǩ�A�*

epsilon�ľ�t�S!.       ��W�	쉉�ǩ�A�* 

Average reward per step�ľ���
       ��2	����ǩ�A�*

epsilon�ľ���l.       ��W�	V���ǩ�A�* 

Average reward per step�ľ�����       ��2	����ǩ�A�*

epsilon�ľ�c<&g.       ��W�	z���ǩ�A�* 

Average reward per step�ľ����       ��2	*��ǩ�A�*

epsilon�ľ�8UU.       ��W�	�|��ǩ�A�* 

Average reward per step�ľ���|       ��2	�}��ǩ�A�*

epsilon�ľ�C�d6.       ��W�	v���ǩ�A�* 

Average reward per step�ľ��~M�       ��2	e���ǩ�A�*

epsilon�ľ����5.       ��W�	kԓ�ǩ�A�* 

Average reward per step�ľ�l�6�       ��2	EՓ�ǩ�A�*

epsilon�ľ��ϡ7.       ��W�	���ǩ�A�* 

Average reward per step�ľ���P�       ��2	}��ǩ�A�*

epsilon�ľ�	oh�.       ��W�	n���ǩ�A�* 

Average reward per step�ľ�-B�       ��2	D���ǩ�A�*

epsilon�ľ�Z�`�.       ��W�	��ǩ�A�* 

Average reward per step�ľ��[�_       ��2	���ǩ�A�*

epsilon�ľ���_=0       ���_	���ǩ�AG*#
!
Average reward per episode�{����C.       ��W�	%��ǩ�AG*!

total reward per episode  ���i.       ��W�	]ܝ�ǩ�A�* 

Average reward per step�{��Nf��       ��2	7ݝ�ǩ�A�*

epsilon�{��nO	b.       ��W�	���ǩ�A�* 

Average reward per step�{�����       ��2	=��ǩ�A�*

epsilon�{��f&�.       ��W�	1C��ǩ�A�* 

Average reward per step�{��`Sx       ��2	�C��ǩ�A�*

epsilon�{���WqB.       ��W�	?���ǩ�A�* 

Average reward per step�{��(;��       ��2	ﭤ�ǩ�A�*

epsilon�{��ꓵ.       ��W�	�)��ǩ�A�* 

Average reward per step�{��Un�       ��2	�*��ǩ�A�*

epsilon�{����Ba.       ��W�	�]��ǩ�A�* 

Average reward per step�{���#`l       ��2	_^��ǩ�A�*

epsilon�{��Z�b.       ��W�	����ǩ�A�* 

Average reward per step�{��g���       ��2	����ǩ�A�*

epsilon�{��C
�.       ��W�	���ǩ�A�* 

Average reward per step�{��1d       ��2	���ǩ�A�*

epsilon�{����	.       ��W�	���ǩ�A�* 

Average reward per step�{���B       ��2	>��ǩ�A�*

epsilon�{��6�r.       ��W�	e���ǩ�A�* 

Average reward per step�{���7Ug       ��2	e���ǩ�A�*

epsilon�{����`.       ��W�	tҲ�ǩ�A�* 

Average reward per step�{���?e5       ��2	 Ӳ�ǩ�A�*

epsilon�{������.       ��W�	���ǩ�A�* 

Average reward per step�{���_`       ��2	���ǩ�A�*

epsilon�{��w�
 .       ��W�	� ��ǩ�A�* 

Average reward per step�{���)�*       ��2	�!��ǩ�A�*

epsilon�{���v�.       ��W�	/���ǩ�A�* 

Average reward per step�{��ZyI7       ��2	���ǩ�A�*

epsilon�{���}�@.       ��W�	;��ǩ�A�* 

Average reward per step�{����;1       ��2	y��ǩ�A�*

epsilon�{��Z��.       ��W�	/0��ǩ�A�* 

Average reward per step�{����f       ��2	�0��ǩ�A�*

epsilon�{���%5.       ��W�	Pƾ�ǩ�A�* 

Average reward per step�{��|���       ��2	*Ǿ�ǩ�A�*

epsilon�{��|l��.       ��W�	���ǩ�A�* 

Average reward per step�{���l       ��2	����ǩ�A�*

epsilon�{���y�.       ��W�	���ǩ�A�* 

Average reward per step�{����я       ��2	���ǩ�A�*

epsilon�{���Y��.       ��W�	-%��ǩ�A�* 

Average reward per step�{��4u��       ��2	1&��ǩ�A�*

epsilon�{���g-%.       ��W�	m���ǩ�A�* 

Average reward per step�{��^��e       ��2	C���ǩ�A�*

epsilon�{����Ƥ.       ��W�	���ǩ�A�* 

Average reward per step�{����7j       ��2	���ǩ�A�*

epsilon�{���z&�.       ��W�	�J��ǩ�A�* 

Average reward per step�{��%`J       ��2	�K��ǩ�A�*

epsilon�{����.       ��W�	�t��ǩ�A�* 

Average reward per step�{���g�       ��2	>u��ǩ�A�*

epsilon�{��P0 .       ��W�	G���ǩ�A�* 

Average reward per step�{��`��O       ��2	i���ǩ�A�*

epsilon�{��zvl.       ��W�	�;��ǩ�A�* 

Average reward per step�{��<��       ��2	h<��ǩ�A�*

epsilon�{��2��.       ��W�	Hl��ǩ�A�* 

Average reward per step�{��&_��       ��2	/m��ǩ�A�*

epsilon�{��>>[�.       ��W�	����ǩ�A�* 

Average reward per step�{����       ��2	_���ǩ�A�*

epsilon�{��B(��.       ��W�	<���ǩ�A�* 

Average reward per step�{��촢�       ��2	+���ǩ�A�*

epsilon�{������.       ��W�	k���ǩ�A�* 

Average reward per step�{��]Y�       ��2	���ǩ�A�*

epsilon�{��.c�.       ��W�	k��ǩ�A�* 

Average reward per step�{��Vj�+       ��2	l��ǩ�A�*

epsilon�{��]|�q.       ��W�	d���ǩ�A�* 

Average reward per step�{�����       ��2	���ǩ�A�*

epsilon�{��QB�S.       ��W�	"���ǩ�A�* 

Average reward per step�{��1+�       ��2	���ǩ�A�*

epsilon�{��P`	.       ��W�	���ǩ�A�* 

Average reward per step�{����       ��2	����ǩ�A�*

epsilon�{��mbCj0       ���_	t���ǩ�AH*#
!
Average reward per episodexxX��.       ��W�	J���ǩ�AH*!

total reward per episode  ��O�.       ��W�	����ǩ�A�* 

Average reward per stepxxX�>���       ��2	����ǩ�A�*

epsilonxxX�F�Yn.       ��W�	:��ǩ�A�* 

Average reward per stepxxX���v�       ��2	���ǩ�A�*

epsilonxxX�X��S.       ��W�	�D��ǩ�A�* 

Average reward per stepxxX�MAx       ��2	|E��ǩ�A�*

epsilonxxX�<�O�.       ��W�	���ǩ�A�* 

Average reward per stepxxX�K��       ��2	����ǩ�A�*

epsilonxxX��;�D.       ��W�	�F��ǩ�A�* 

Average reward per stepxxX���{       ��2	�G��ǩ�A�*

epsilonxxX�����.       ��W�	�f��ǩ�A�* 

Average reward per stepxxX�����       ��2	og��ǩ�A�*

epsilonxxX��旤.       ��W�	�X��ǩ�A�* 

Average reward per stepxxX�'�%�       ��2	OY��ǩ�A�*

epsilonxxX�(Ѯ.       ��W�	�{��ǩ�A�* 

Average reward per stepxxX�ޯ�W       ��2	�|��ǩ�A�*

epsilonxxX�\��.       ��W�	���ǩ�A�* 

Average reward per stepxxX��<`.       ��2	����ǩ�A�*

epsilonxxX����.       ��W�	)��ǩ�A�* 

Average reward per stepxxX��Xߤ       ��2	���ǩ�A�*

epsilonxxX��m8�.       ��W�	k���ǩ�A�* 

Average reward per stepxxX��4       ��2	f���ǩ�A�*

epsilonxxX�Q�/.       ��W�	j� �ǩ�A�* 

Average reward per stepxxX���$       ��2	� �ǩ�A�*

epsilonxxX���T�.       ��W�	�2�ǩ�A�* 

Average reward per stepxxX�,w�(       ��2	r3�ǩ�A�*

epsilonxxX��	7�.       ��W�	���ǩ�A�* 

Average reward per stepxxX��N�5       ��2	C��ǩ�A�*

epsilonxxX����7.       ��W�	x�ǩ�A�* 

Average reward per stepxxX�f�4�       ��2	=�ǩ�A�*

epsilonxxX��x�.       ��W�	�=	�ǩ�A�* 

Average reward per stepxxX�����       ��2	d>	�ǩ�A�*

epsilonxxX�'Z�.       ��W�	���ǩ�A�* 

Average reward per stepxxX�v��       ��2	W��ǩ�A�*

epsilonxxX��Z�.       ��W�	��ǩ�A�* 

Average reward per stepxxX��m�       ��2	� �ǩ�A�*

epsilonxxX�녏�.       ��W�	x��ǩ�A�* 

Average reward per stepxxX�x���       ��2	c��ǩ�A�*

epsilonxxX�`�.       ��W�	�t�ǩ�A�* 

Average reward per stepxxX�/�]�       ��2	yu�ǩ�A�*

epsilonxxX�C<ڭ.       ��W�	m��ǩ�A�* 

Average reward per stepxxX��?X#       ��2	��ǩ�A�*

epsilonxxX���!�.       ��W�	Y��ǩ�A�* 

Average reward per stepxxX�2M@�       ��2	@��ǩ�A�*

epsilonxxX� �I+.       ��W�	���ǩ�A�* 

Average reward per stepxxX��u�       ��2	ǝ�ǩ�A�*

epsilonxxX��/�.       ��W�	��ǩ�A�* 

Average reward per stepxxX��F��       ��2	���ǩ�A�*

epsilonxxX� PX�.       ��W�	��ǩ�A�* 

Average reward per stepxxX��X�a       ��2	��ǩ�A�*

epsilonxxX���ZZ.       ��W�	���ǩ�A�* 

Average reward per stepxxX��C9P       ��2	n��ǩ�A�*

epsilonxxX�����.       ��W�	غ�ǩ�A�* 

Average reward per stepxxX�U��        ��2	˻�ǩ�A�*

epsilonxxX���׶.       ��W�	v�!�ǩ�A�* 

Average reward per stepxxX�.VP       ��2	q�!�ǩ�A�*

epsilonxxX���.       ��W�	1^$�ǩ�A�* 

Average reward per stepxxX�J*�u       ��2	�^$�ǩ�A�*

epsilonxxX�¤H�.       ��W�	��%�ǩ�A�* 

Average reward per stepxxX�K��       ��2	g�%�ǩ�A�*

epsilonxxX���S.       ��W�	�3(�ǩ�A�* 

Average reward per stepxxX�z9E       ��2	�4(�ǩ�A�*

epsilonxxX�Rho�0       ���_	�c(�ǩ�AI*#
!
Average reward per episode�s���"+�.       ��W�	�d(�ǩ�AI*!

total reward per episode  
�2�.       ��W�	�,�ǩ�A�* 

Average reward per step�s��γ��       ��2	e,�ǩ�A�*

epsilon�s�����.       ��W�	%X.�ǩ�A�* 

Average reward per step�s����9}       ��2	�X.�ǩ�A�*

epsilon�s��>ƵT.       ��W�	��0�ǩ�A�* 

Average reward per step�s��s�I�       ��2	S�0�ǩ�A�*

epsilon�s���MJ�.       ��W�	R3�ǩ�A�* 

Average reward per step�s��U�t�       ��2	A3�ǩ�A�*

epsilon�s��h8v.       ��W�	Ӈ4�ǩ�A�* 

Average reward per step�s��U��       ��2	��4�ǩ�A�*

epsilon�s��Д��.       ��W�	@M6�ǩ�A�* 

Average reward per step�s���wi       ��2	N6�ǩ�A�*

epsilon�s����.       ��W�	�8�ǩ�A�* 

Average reward per step�s���x2�       ��2	��8�ǩ�A�*

epsilon�s���z�.       ��W�	��:�ǩ�A�* 

Average reward per step�s��Jo��       ��2	e�:�ǩ�A�*

epsilon�s��Y�*�.       ��W�	0=�ǩ�A�* 

Average reward per step�s��U\       ��2	�0=�ǩ�A�*

epsilon�s���pY.       ��W�	~�>�ǩ�A�* 

Average reward per step�s��M=��       ��2	.�>�ǩ�A�*

epsilon�s���.       ��W�	��@�ǩ�A�* 

Average reward per step�s��.�g�       ��2	m�@�ǩ�A�*

epsilon�s��ܟ��.       ��W�	)BC�ǩ�A�* 

Average reward per step�s��9
D�       ��2	�BC�ǩ�A�*

epsilon�s��E�z�.       ��W�	��D�ǩ�A�* 

Average reward per step�s������       ��2	1�D�ǩ�A�*

epsilon�s���o.       ��W�	��F�ǩ�A�* 

Average reward per step�s��;[B       ��2	��F�ǩ�A�*

epsilon�s��E=�.       ��W�	�H�ǩ�A�* 

Average reward per step�s������       ��2	��H�ǩ�A�*

epsilon�s��_.       ��W�	HK�ǩ�A�* 

Average reward per step�s��ڈ��       ��2	�K�ǩ�A�*

epsilon�s���_��.       ��W�	�tM�ǩ�A�* 

Average reward per step�s��ʀ�       ��2	yuM�ǩ�A�*

epsilon�s��v��@.       ��W�	+�O�ǩ�A�* 

Average reward per step�s����L�       ��2	ӡO�ǩ�A�*

epsilon�s��B�ʥ.       ��W�	"Q�ǩ�A�* 

Average reward per step�s�����X       ��2	�"Q�ǩ�A�*

epsilon�s��'���.       ��W�	YLS�ǩ�A�* 

Average reward per step�s���u�@       ��2	4MS�ǩ�A�*

epsilon�s��O��.       ��W�	ۦU�ǩ�A�* 

Average reward per step�s���?��       ��2	z�U�ǩ�A�*

epsilon�s�����9.       ��W�		�W�ǩ�A�* 

Average reward per step�s���}�       ��2	��W�ǩ�A�*

epsilon�s��F�U.       ��W�	��Y�ǩ�A�* 

Average reward per step�s��t�y�       ��2	��Y�ǩ�A�*

epsilon�s��9��.       ��W�	�\�ǩ�A�* 

Average reward per step�s��
0U6       ��2	�\�ǩ�A�*

epsilon�s���r^.       ��W�	QJ^�ǩ�A�* 

Average reward per step�s��P�n       ��2	K^�ǩ�A�*

epsilon�s���6dk.       ��W�	�`�ǩ�A�* 

Average reward per step�s��7އ�       ��2	�`�ǩ�A�*

epsilon�s���T�.       ��W�	�Eb�ǩ�A�* 

Average reward per step�s��9��I       ��2	xFb�ǩ�A�*

epsilon�s���k�V.       ��W�	�[d�ǩ�A�* 

Average reward per step�s��=:Y�       ��2	�\d�ǩ�A�*

epsilon�s��˘.       ��W�	\�e�ǩ�A�* 

Average reward per step�s���z'       ��2	��e�ǩ�A�*

epsilon�s��z��u.       ��W�	h�ǩ�A�* 

Average reward per step�s��90��       ��2	�h�ǩ�A�*

epsilon�s���B.       ��W�	Jj�ǩ�A�* 

Average reward per step�s��a�/�       ��2	�Jj�ǩ�A�*

epsilon�s��Rhu�.       ��W�	 bl�ǩ�A�* 

Average reward per step�s��Ƣ�       ��2	cl�ǩ�A�*

epsilon�s����{�.       ��W�	��m�ǩ�A�* 

Average reward per step�s��d�        ��2	��m�ǩ�A�*

epsilon�s��G���.       ��W�	^p�ǩ�A�* 

Average reward per step�s���[       ��2	<p�ǩ�A�*

epsilon�s��%���.       ��W�	8Nr�ǩ�A�* 

Average reward per step�s�� �X�       ��2	�Nr�ǩ�A�*

epsilon�s����T;.       ��W�	�`t�ǩ�A�* 

Average reward per step�s���zd        ��2	�at�ǩ�A�*

epsilon�s��$8��.       ��W�	 v�ǩ�A�* 

Average reward per step�s��v�_       ��2	�v�ǩ�A�*

epsilon�s���z��.       ��W�	�x�ǩ�A�* 

Average reward per step�s��{�@Q       ��2	�x�ǩ�A�*

epsilon�s�����.       ��W�	e�z�ǩ�A�* 

Average reward per step�s����6       ��2	 �z�ǩ�A�*

epsilon�s��݊�.       ��W�	�|�ǩ�A�* 

Average reward per step�s�����       ��2	��|�ǩ�A�*

epsilon�s������.       ��W�	t�ǩ�A�* 

Average reward per step�s��.�M=       ��2	J	�ǩ�A�*

epsilon�s���h�E.       ��W�	z���ǩ�A�* 

Average reward per step�s��L;�       ��2	H���ǩ�A�*

epsilon�s���9�.       ��W�	F���ǩ�A�* 

Average reward per step�s�����i       ��2	x���ǩ�A�*

epsilon�s��`e.       ��W�	o��ǩ�A�* 

Average reward per step�s��1rY�       ��2	E���ǩ�A�*

epsilon�s��9���.       ��W�	�E��ǩ�A�* 

Average reward per step�s��)�)�       ��2	�F��ǩ�A�*

epsilon�s���C��.       ��W�	Mڈ�ǩ�A�* 

Average reward per step�s��<B�       ��2	#ۈ�ǩ�A�*

epsilon�s��^qL�.       ��W�	����ǩ�A�* 

Average reward per step�s���"i�       ��2	v���ǩ�A�*

epsilon�s���9��.       ��W�	b��ǩ�A�* 

Average reward per step�s���p�y       ��2	�b��ǩ�A�*

epsilon�s���ZÈ.       ��W�	���ǩ�A�* 

Average reward per step�s��S9%       ��2	j��ǩ�A�*

epsilon�s�����..       ��W�	�R��ǩ�A�* 

Average reward per step�s��z�w       ��2	TS��ǩ�A�*

epsilon�s��@�Q.       ��W�	,f��ǩ�A�* 

Average reward per step�s����'�       ��2	�f��ǩ�A�*

epsilon�s��� ��.       ��W�	���ǩ�A�* 

Average reward per step�s����       ��2	����ǩ�A�*

epsilon�s��K��.       ��W�	���ǩ�A�* 

Average reward per step�s����da       ��2	4��ǩ�A�*

epsilon�s�����/.       ��W�	{���ǩ�A�* 

Average reward per step�s��\;X       ��2	]���ǩ�A�*

epsilon�s�����+.       ��W�	mǚ�ǩ�A�* 

Average reward per step�s��KQ#       ��2	Ț�ǩ�A�*

epsilon�s��ò�.       ��W�	����ǩ�A�* 

Average reward per step�s��ם       ��2	���ǩ�A�*

epsilon�s��%�3.       ��W�	�Z��ǩ�A�* 

Average reward per step�s�����       ��2	�[��ǩ�A�*

epsilon�s����J.       ��W�	A؟�ǩ�A�* 

Average reward per step�s���?       ��2	ٟ�ǩ�A�*

epsilon�s���W�.       ��W�	��ǩ�A�* 

Average reward per step�s��*g%�       ��2	���ǩ�A�*

epsilon�s������.       ��W�	40��ǩ�A�* 

Average reward per step�s�����        ��2	�0��ǩ�A�*

epsilon�s���u�E.       ��W�	����ǩ�A�* 

Average reward per step�s��+'3�       ��2	����ǩ�A�*

epsilon�s���`�h.       ��W�	�ܧ�ǩ�A�* 

Average reward per step�s���`]       ��2	ݧ�ǩ�A�*

epsilon�s������.       ��W�	����ǩ�A�* 

Average reward per step�s�����6       ��2	L���ǩ�A�*

epsilon�s���L��.       ��W�	~8��ǩ�A�* 

Average reward per step�s��Ȓ<k       ��2	79��ǩ�A�*

epsilon�s���mɕ.       ��W�	#���ǩ�A�* 

Average reward per step�s�����C       ��2	����ǩ�A�*

epsilon�s����<.       ��W�	�J��ǩ�A�* 

Average reward per step�s��ʝ�<       ��2	wK��ǩ�A�*

epsilon�s���}.       ��W�	�t��ǩ�A�* 

Average reward per step�s����/4       ��2	�u��ǩ�A�*

epsilon�s��m�]R.       ��W�	ܷ��ǩ�A�* 

Average reward per step�s��l��       ��2	����ǩ�A�*

epsilon�s���s��.       ��W�	�,��ǩ�A�* 

Average reward per step�s��7/�       ��2	�-��ǩ�A�*

epsilon�s��	lآ.       ��W�	�a��ǩ�A�* 

Average reward per step�s������       ��2	5b��ǩ�A�*

epsilon�s��J'gB.       ��W�	.Ƽ�ǩ�A�* 

Average reward per step�s��,�J       ��2	*Ǽ�ǩ�A�*

epsilon�s��酮.       ��W�	U��ǩ�A�* 

Average reward per step�s�����       ��2	�U��ǩ�A�*

epsilon�s���5�.       ��W�	R}��ǩ�A�* 

Average reward per step�s��$�P(       ��2	R~��ǩ�A�*

epsilon�s�����.       ��W�	Y���ǩ�A�* 

Average reward per step�s��i�`       ��2	;���ǩ�A�*

epsilon�s���W�.       ��W�	���ǩ�A�* 

Average reward per step�s��c�A�       ��2	��ǩ�A�*

epsilon�s��+�)�.       ��W�	K���ǩ�A�* 

Average reward per step�s��v��U       ��2	���ǩ�A�*

epsilon�s�����.       ��W�	M���ǩ�A�* 

Average reward per step�s�����       ��2	���ǩ�A�*

epsilon�s����B�.       ��W�	����ǩ�A�* 

Average reward per step�s����_       ��2	R���ǩ�A�*

epsilon�s����uD.       ��W�	��ǩ�A�* 

Average reward per step�s��L�	+       ��2	���ǩ�A�*

epsilon�s��f�P.       ��W�	L��ǩ�A�* 

Average reward per step�s����j       ��2	�L��ǩ�A�*

epsilon�s��z�J.       ��W�	#���ǩ�A�* 

Average reward per step�s����G       ��2	����ǩ�A�*

epsilon�s���;��.       ��W�	���ǩ�A�* 

Average reward per step�s��җn)       ��2	���ǩ�A�*

epsilon�s���Ηk.       ��W�	p��ǩ�A�* 

Average reward per step�s�����5       ��2	�p��ǩ�A�*

epsilon�s����-.       ��W�	S!��ǩ�A�* 

Average reward per step�s��]{��       ��2	)"��ǩ�A�*

epsilon�s���0}q.       ��W�	*U��ǩ�A�* 

Average reward per step�s��2�Vs       ��2	�U��ǩ�A�*

epsilon�s���^�t.       ��W�	}��ǩ�A�* 

Average reward per step�s��^|��       ��2	�}��ǩ�A�*

epsilon�s���r�0       ���_	Ǜ��ǩ�AJ*#
!
Average reward per episode�;b����.       ��W�	V���ǩ�AJ*!

total reward per episode  �����.       ��W�	�f��ǩ�A�* 

Average reward per step�;b��6       ��2	�g��ǩ�A�*

epsilon�;b�����.       ��W�	&���ǩ�A�* 

Average reward per step�;b�q{�S       ��2	����ǩ�A�*

epsilon�;b��+ؼ.       ��W�	���ǩ�A�* 

Average reward per step�;b���ɪ       ��2	����ǩ�A�*

epsilon�;b� _�.       ��W�	���ǩ�A�* 

Average reward per step�;b�O�2r       ��2	����ǩ�A�*

epsilon�;b��.�G.       ��W�	L���ǩ�A�* 

Average reward per step�;b�/�n�       ��2	����ǩ�A�*

epsilon�;b�?��y.       ��W�	r��ǩ�A�* 

Average reward per step�;b�}p��       ��2	�r��ǩ�A�*

epsilon�;b���.       ��W�	���ǩ�A�* 

Average reward per step�;b�r��       ��2	���ǩ�A�*

epsilon�;b�A�c'.       ��W�	����ǩ�A�* 

Average reward per step�;b�#�	o       ��2	����ǩ�A�*

epsilon�;b�i���.       ��W�	�b��ǩ�A�* 

Average reward per step�;b��7�       ��2	$c��ǩ�A�*

epsilon�;b����.       ��W�	���ǩ�A�* 

Average reward per step�;b�o�hp       ��2	����ǩ�A�*

epsilon�;b�YOY.       ��W�	���ǩ�A�* 

Average reward per step�;b��ܪ       ��2	7��ǩ�A�*

epsilon�;b�+˚/.       ��W�	x*��ǩ�A�* 

Average reward per step�;b���M       ��2	(+��ǩ�A�*

epsilon�;b�Q��x.       ��W�	VH��ǩ�A�* 

Average reward per step�;b����b       ��2	0I��ǩ�A�*

epsilon�;b�D�.       ��W�	����ǩ�A�* 

Average reward per step�;b�8]��       ��2	t���ǩ�A�*

epsilon�;b��h&-.       ��W�	g~��ǩ�A�* 

Average reward per step�;b�P�\L       ��2	��ǩ�A�*

epsilon�;b�~t��.       ��W�	h ��ǩ�A�* 

Average reward per step�;b�Ni�       ��2	)!��ǩ�A�*

epsilon�;b��*.       ��W�	C= �ǩ�A�* 

Average reward per step�;b�� ��       ��2	�= �ǩ�A�*

epsilon�;b�1�h.       ��W�	T��ǩ�A�* 

Average reward per step�;b�=��       ��2	*��ǩ�A�*

epsilon�;b�f�W<.       ��W�	X �ǩ�A�* 

Average reward per step�;b�g	x�       ��2	� �ǩ�A�*

epsilon�;b�E�G.       ��W�	�"�ǩ�A�* 

Average reward per step�;b�+       ��2	F#�ǩ�A�*

epsilon�;b��c�.       ��W�	�<�ǩ�A�* 

Average reward per step�;b�1ck       ��2	�=�ǩ�A�*

epsilon�;b��@��.       ��W�	�i
�ǩ�A�* 

Average reward per step�;b���5       ��2	rj
�ǩ�A�*

epsilon�;b��Y�.       ��W�	���ǩ�A�* 

Average reward per step�;b���4       ��2	w��ǩ�A�*

epsilon�;b��:T.       ��W�	���ǩ�A�* 

Average reward per step�;b�kj�       ��2	���ǩ�A�*

epsilon�;b�6�k�.       ��W�	�ǩ�A�* 

Average reward per step�;b��;�       ��2	��ǩ�A�*

epsilon�;b���.       ��W�	���ǩ�A�* 

Average reward per step�;b��R��       ��2	��ǩ�A�*

epsilon�;b�N�f.       ��W�	���ǩ�A�* 

Average reward per step�;b��>��       ��2	9��ǩ�A�*

epsilon�;b�=3�*.       ��W�	D��ǩ�A�* 

Average reward per step�;b��-ڹ       ��2	���ǩ�A�*

epsilon�;b� ��.       ��W�	/��ǩ�A�* 

Average reward per step�;b���h>       ��2		��ǩ�A�*

epsilon�;b�Ө�?.       ��W�	F��ǩ�A�* 

Average reward per step�;b���8�       ��2	,��ǩ�A�*

epsilon�;b��_[.       ��W�	C�ǩ�A�* 

Average reward per step�;b�!=��       ��2	��ǩ�A�*

epsilon�;b�Ijα.       ��W�	H�ǩ�A�* 

Average reward per step�;b��	�
       ��2	�H�ǩ�A�*

epsilon�;b�"�y�.       ��W�	�	!�ǩ�A�* 

Average reward per step�;b�
�v�       ��2	�
!�ǩ�A�*

epsilon�;b��i^:.       ��W�	�C#�ǩ�A�* 

Average reward per step�;b����[       ��2	�D#�ǩ�A�*

epsilon�;b��dt�.       ��W�	l�$�ǩ�A�* 

Average reward per step�;b�Q��d       ��2	5�$�ǩ�A�*

epsilon�;b�&���.       ��W�	��&�ǩ�A�* 

Average reward per step�;b�� �K       ��2	o�&�ǩ�A�*

epsilon�;b���e{.       ��W�	�)�ǩ�A�* 

Average reward per step�;b���6s       ��2	/)�ǩ�A�*

epsilon�;b���[�.       ��W�	�G+�ǩ�A�* 

Average reward per step�;b�X�;�       ��2	�H+�ǩ�A�*

epsilon�;b�J��.       ��W�	`Y-�ǩ�A�* 

Average reward per step�;b��h�       ��2	:Z-�ǩ�A�*

epsilon�;b�5z��.       ��W�	'�/�ǩ�A�* 

Average reward per step�;b���J       ��2	�/�ǩ�A�*

epsilon�;b�����.       ��W�	<L1�ǩ�A�* 

Average reward per step�;b�f��       ��2	#M1�ǩ�A�*

epsilon�;b�].�f.       ��W�	Q�3�ǩ�A�* 

Average reward per step�;b��2Wf       ��2	�3�ǩ�A�*

epsilon�;b��.��0       ���_	ϡ3�ǩ�AK*#
!
Average reward per episode�</�N��.       ��W�	U�3�ǩ�AK*!

total reward per episode  ��(��/.       ��W�	�9�ǩ�A�* 

Average reward per step�</�75�u       ��2	��9�ǩ�A�*

epsilon�</�!�#8.       ��W�	��;�ǩ�A�* 

Average reward per step�</�^�$�       ��2	��;�ǩ�A�*

epsilon�</�ܾ�<.       ��W�	�/>�ǩ�A�* 

Average reward per step�</�nA�       ��2	n0>�ǩ�A�*

epsilon�</���`�.       ��W�	��?�ǩ�A�* 

Average reward per step�</���%       ��2	F�?�ǩ�A�*

epsilon�</�&U@.       ��W�	L�A�ǩ�A�* 

Average reward per step�</����       ��2	�A�ǩ�A�*

epsilon�</��CR.       ��W�	�D�ǩ�A�* 

Average reward per step�</�{��       ��2	�D�ǩ�A�*

epsilon�</��tFo.       ��W�	�@F�ǩ�A�* 

Average reward per step�</�.Ҋ�       ��2	�AF�ǩ�A�*

epsilon�</�܆�S.       ��W�	
�G�ǩ�A�* 

Average reward per step�</��ӫ       ��2	��G�ǩ�A�*

epsilon�</�@�%�.       ��W�	#�I�ǩ�A�* 

Average reward per step�</�@��	       ��2	��I�ǩ�A�*

epsilon�</�"���.       ��W�	�L�ǩ�A�* 

Average reward per step�</��Ջ�       ��2	�L�ǩ�A�*

epsilon�</�����.       ��W�	!:N�ǩ�A�* 

Average reward per step�</�j@e       ��2	�:N�ǩ�A�*

epsilon�</�	��v.       ��W�	��P�ǩ�A�* 

Average reward per step�</��Nz       ��2	��P�ǩ�A�*

epsilon�</�h,��.       ��W�	sKR�ǩ�A�* 

Average reward per step�</�OV       ��2	LR�ǩ�A�*

epsilon�</��* .       ��W�	eoT�ǩ�A�* 

Average reward per step�</�K
��       ��2		pT�ǩ�A�*

epsilon�</�`ts�.       ��W�	�V�ǩ�A�* 

Average reward per step�</���R       ��2	t�V�ǩ�A�*

epsilon�</�T���.       ��W�	��X�ǩ�A�* 

Average reward per step�</��s�V       ��2	i�X�ǩ�A�*

epsilon�</�BE�}.       ��W�	ePZ�ǩ�A�* 

Average reward per step�</�?�R)       ��2	7QZ�ǩ�A�*

epsilon�</�6�[.       ��W�	�v\�ǩ�A�* 

Average reward per step�</�Ǵqt       ��2	Gw\�ǩ�A�*

epsilon�</��Y��.       ��W�	��^�ǩ�A�* 

Average reward per step�</�S��       ��2	q�^�ǩ�A�*

epsilon�</�GIθ.       ��W�	E�`�ǩ�A�* 

Average reward per step�</�%�O�       ��2	��`�ǩ�A�*

epsilon�</���}.       ��W�	�jc�ǩ�A�* 

Average reward per step�</�pO�       ��2	kc�ǩ�A�*

epsilon�</���.       ��W�	��f�ǩ�A�* 

Average reward per step�</����z       ��2	�f�ǩ�A�*

epsilon�</�Ɩ�}.       ��W�	l�h�ǩ�A�* 

Average reward per step�</�M]8z       ��2	�h�ǩ�A�*

epsilon�</�я:�.       ��W�	�k�ǩ�A�* 

Average reward per step�</���       ��2	�k�ǩ�A�*

epsilon�</�j���.       ��W�	s,m�ǩ�A�* 

Average reward per step�</��v�       ��2	A-m�ǩ�A�*

epsilon�</��4�.       ��W�	�n�ǩ�A�* 

Average reward per step�</��k|�       ��2	˜n�ǩ�A�*

epsilon�</�I���.       ��W�	
�p�ǩ�A�* 

Average reward per step�</�!6��       ��2	��p�ǩ�A�*

epsilon�</���K.       ��W�	�s�ǩ�A�* 

Average reward per step�</�.�M�       ��2	\s�ǩ�A�*

epsilon�</����.       ��W�	Yu�ǩ�A�* 

Average reward per step�</�WV�
       ��2	3u�ǩ�A�*

epsilon�</���A.       ��W�	I-w�ǩ�A�* 

Average reward per step�</�6��       ��2	/w�ǩ�A�*

epsilon�</�A���.       ��W�	`=y�ǩ�A�* 

Average reward per step�</�څ�-       ��2	�>y�ǩ�A�*

epsilon�</�б:�.       ��W�	G{�ǩ�A�* 

Average reward per step�</�[mw       ��2	�G{�ǩ�A�*

epsilon�</�@@l
.       ��W�	eq}�ǩ�A�* 

Average reward per step�</��!�K       ��2	�q}�ǩ�A�*

epsilon�</��C�5.       ��W�	̛�ǩ�A�* 

Average reward per step�</�o��o       ��2	^��ǩ�A�*

epsilon�</��҄V.       ��W�	�́�ǩ�A�* 

Average reward per step�</��h8�       ��2	:΁�ǩ�A�*

epsilon�</�	� .       ��W�	p^��ǩ�A�* 

Average reward per step�</�lo`�       ��2	|_��ǩ�A�*

epsilon�</�ټ�t.       ��W�	����ǩ�A�* 

Average reward per step�</�p4       ��2	ƥ��ǩ�A�*

epsilon�</�uAT.       ��W�	?ȇ�ǩ�A�* 

Average reward per step�</�Ҥ�       ��2	ɇ�ǩ�A�*

epsilon�</��h�.       ��W�	B��ǩ�A�* 

Average reward per step�</�)��       ��2	��ǩ�A�*

epsilon�</��BB.       ��W�	��ǩ�A�* 

Average reward per step�</��P�       ��2	���ǩ�A�*

epsilon�</�[{fn.       ��W�	�-��ǩ�A�* 

Average reward per step�</�g6�       ��2	w.��ǩ�A�*

epsilon�</�2�4.       ��W�	����ǩ�A�* 

Average reward per step�</���Ԍ       ��2	����ǩ�A�*

epsilon�</��`�o.       ��W�	5��ǩ�A�* 

Average reward per step�</�me��       ��2	���ǩ�A�*

epsilon�</�4�Qj.       ��W�	#��ǩ�A�* 

Average reward per step�</�ˉM       ��2	�#��ǩ�A�*

epsilon�</�kP<�.       ��W�	E��ǩ�A�* 

Average reward per step�</���K�       ��2	�E��ǩ�A�*

epsilon�</�0��
.       ��W�	�b��ǩ�A�* 

Average reward per step�</�,|�f       ��2	�c��ǩ�A�*

epsilon�</���!�.       ��W�	Uٙ�ǩ�A�* 

Average reward per step�</��ӆ       ��2	ڙ�ǩ�A�*

epsilon�</�P�,.       ��W�	���ǩ�A�* 

Average reward per step�</�%��       ��2	u��ǩ�A�*

epsilon�</�G��.       ��W�	5��ǩ�A�* 

Average reward per step�</�Ӑ�       ��2	�5��ǩ�A�*

epsilon�</��hi.       ��W�	'l��ǩ�A�* 

Average reward per step�</�}t�       ��2	�l��ǩ�A�*

epsilon�</��$/3.       ��W�	���ǩ�A�* 

Average reward per step�</�0���       ��2	���ǩ�A�*

epsilon�</�퀐�.       ��W�	����ǩ�A�* 

Average reward per step�</����]       ��2	t���ǩ�A�*

epsilon�</�i�.       ��W�	�ئ�ǩ�A�* 

Average reward per step�</��b�       ��2	�٦�ǩ�A�*

epsilon�</����q.       ��W�	zQ��ǩ�A�* 

Average reward per step�</��i��       ��2	"R��ǩ�A�*

epsilon�</��類.       ��W�	H���ǩ�A�* 

Average reward per step�</�(N�       ��2	쇪�ǩ�A�*

epsilon�</���.       ��W�	�2��ǩ�A�* 

Average reward per step�</�R�d�       ��2	L3��ǩ�A�*

epsilon�</�uF��.       ��W�	dY��ǩ�A�* 

Average reward per step�</��8@�       ��2	:Z��ǩ�A�*

epsilon�</�G�T�.       ��W�	�+��ǩ�A�* 

Average reward per step�</�M6       ��2	-��ǩ�A�*

epsilon�</�����.       ��W�	���ǩ�A�* 

Average reward per step�</�2y6v       ��2	פ��ǩ�A�*

epsilon�</�|��~.       ��W�	$a��ǩ�A�* 

Average reward per step�</����>       ��2	9b��ǩ�A�*

epsilon�</���w1.       ��W�	� ��ǩ�A�* 

Average reward per step�</�y�5       ��2	���ǩ�A�*

epsilon�</�_>�c.       ��W�	��ǩ�A�* 

Average reward per step�</��C�9       ��2	���ǩ�A�*

epsilon�</�X=b.       ��W�	���ǩ�A�* 

Average reward per step�</���8�       ��2	��ǩ�A�*

epsilon�</����.       ��W�	����ǩ�A�* 

Average reward per step�</����       ��2	����ǩ�A�*

epsilon�</�e�0].       ��W�	E��ǩ�A�* 

Average reward per step�</�;7        ��2	��ǩ�A�*

epsilon�</�V��.       ��W�	���ǩ�A�* 

Average reward per step�</����O       ��2	���ǩ�A�*

epsilon�</��jy�.       ��W�	!��ǩ�A�* 

Average reward per step�</��H�%       ��2	�!��ǩ�A�*

epsilon�</�ʇ�3.       ��W�	@��ǩ�A�* 

Average reward per step�</��P�       ��2	�@��ǩ�A�*

epsilon�</��`Tf.       ��W�	Um��ǩ�A�* 

Average reward per step�</��U�C       ��2	n��ǩ�A�*

epsilon�</��ox�.       ��W�	���ǩ�A�* 

Average reward per step�</���       ��2	ݖ��ǩ�A�*

epsilon�</�u�6.       ��W�	����ǩ�A�* 

Average reward per step�</�����       ��2	]���ǩ�A�*

epsilon�</�m>�&0       ���_	����ǩ�AL*#
!
Average reward per episode�	�pFbY.       ��W�	/���ǩ�AL*!

total reward per episode  �<Ҵ.       ��W�	~t��ǩ�A�* 

Average reward per step�	�캒�       ��2	Su��ǩ�A�*

epsilon�	�D/J.       ��W�	q���ǩ�A�* 

Average reward per step�	�4a��       ��2	T���ǩ�A�*

epsilon�	��t�.       ��W�	s��ǩ�A�* 

Average reward per step�	����       ��2	�s��ǩ�A�*

epsilon�	�q���.       ��W�	K���ǩ�A�* 

Average reward per step�	�h��       ��2	ޏ��ǩ�A�*

epsilon�	�`��.       ��W�	����ǩ�A�* 

Average reward per step�	��ݑ       ��2	[���ǩ�A�*

epsilon�	�	M��.       ��W�	����ǩ�A�* 

Average reward per step�	�n�       ��2	����ǩ�A�*

epsilon�	�U�1.       ��W�	F(��ǩ�A�* 

Average reward per step�	���^F       ��2	�(��ǩ�A�*

epsilon�	�����.       ��W�	�D��ǩ�A�* 

Average reward per step�	�?D��       ��2	NE��ǩ�A�*

epsilon�	�~C.       ��W�	����ǩ�A�* 

Average reward per step�	�;NT�       ��2	d���ǩ�A�*

epsilon�	�y~%.       ��W�	����ǩ�A�* 

Average reward per step�	�����       ��2	����ǩ�A�*

epsilon�	�+��@.       ��W�	���ǩ�A�* 

Average reward per step�	�C��q       ��2	���ǩ�A�*

epsilon�	��J�7.       ��W�	7�'�ǩ�A�* 

Average reward per step�	��.��       ��2	��'�ǩ�A�*

epsilon�	�Id�.       ��W�	�3*�ǩ�A�* 

Average reward per step�	�$T�       ��2	r4*�ǩ�A�*

epsilon�	��y.       ��W�	�_,�ǩ�A�* 

Average reward per step�	���ܼ       ��2	g`,�ǩ�A�*

epsilon�	�����.       ��W�	��.�ǩ�A�* 

Average reward per step�	�$c�       ��2	Y�.�ǩ�A�*

epsilon�	�E�].       ��W�	��0�ǩ�A�* 

Average reward per step�	�x]?8       ��2	U�0�ǩ�A�*

epsilon�	� )��.       ��W�	��2�ǩ�A�* 

Average reward per step�	����       ��2	�2�ǩ�A�*

epsilon�	�n�gf.       ��W�	�^4�ǩ�A�* 

Average reward per step�	��j>�       ��2	g_4�ǩ�A�*

epsilon�	���5v.       ��W�	��6�ǩ�A�* 

Average reward per step�	����       ��2	:�6�ǩ�A�*

epsilon�	�L-Q�.       ��W�	/�8�ǩ�A�* 

Average reward per step�	�]�i�       ��2	��8�ǩ�A�*

epsilon�	��B��0       ���_	�9�ǩ�AM*#
!
Average reward per episode����1���.       ��W�	9�ǩ�AM*!

total reward per episode  �d���.       ��W�	��<�ǩ�A�* 

Average reward per step����V&VZ       ��2	o�<�ǩ�A�*

epsilon����#�$.       ��W�	�?�ǩ�A�* 

Average reward per step����BD��       ��2	}?�ǩ�A�*

epsilon�������.       ��W�	uA�ǩ�A�* 

Average reward per step����g��       ��2	FA�ǩ�A�*

epsilon���� ��{.       ��W�	�C�ǩ�A�* 

Average reward per step������,m       ��2	C�ǩ�A�*

epsilon�����G��.       ��W�	2E�ǩ�A�* 

Average reward per step����:���       ��2	�2E�ǩ�A�*

epsilon����,��_.       ��W�	�^G�ǩ�A�* 

Average reward per step�����ȹ�       ��2	J_G�ǩ�A�*

epsilon����P�$.       ��W�	�oI�ǩ�A�* 

Average reward per step�������       ��2	�pI�ǩ�A�*

epsilon�����S.       ��W�	�K�ǩ�A�* 

Average reward per step����?�?-       ��2	��K�ǩ�A�*

epsilon������l.       ��W�	�M�ǩ�A�* 

Average reward per step�����3?       ��2	9�M�ǩ�A�*

epsilon����0i�s.       ��W�	-O�ǩ�A�* 

Average reward per step����?�L       ��2	�-O�ǩ�A�*

epsilon����:�W�.       ��W�	=�Q�ǩ�A�* 

Average reward per step����5g�Z       ��2	�Q�ǩ�A�*

epsilon�����|.       ��W�	K!T�ǩ�A�* 

Average reward per step����
�F�       ��2	�!T�ǩ�A�*

epsilon����o�.       ��W�	y�U�ǩ�A�* 

Average reward per step�����Om       ��2	��U�ǩ�A�*

epsilon����~o>�.       ��W�	X�ǩ�A�* 

Average reward per step����)���       ��2	�X�ǩ�A�*

epsilon����Q6h�.       ��W�	�Z�ǩ�A�* 

Average reward per step����p��       ��2	�Z�ǩ�A�*

epsilon����0�(.       ��W�	�[�ǩ�A�* 

Average reward per step�����B6�       ��2	��[�ǩ�A�*

epsilon������X.       ��W�	�]�ǩ�A�* 

Average reward per step������v       ��2	v]�ǩ�A�*

epsilon��������.       ��W�	�*_�ǩ�A�* 

Average reward per step�������       ��2	o+_�ǩ�A�*

epsilon����:~S/.       ��W�	�Fa�ǩ�A�* 

Average reward per step����Ĥu�       ��2	cGa�ǩ�A�*

epsilon����@�f�.       ��W�	2qc�ǩ�A�* 

Average reward per step����@Y       ��2	&rc�ǩ�A�*

epsilon����9��.       ��W�	��d�ǩ�A�* 

Average reward per step�����;�       ��2	7�d�ǩ�A�*

epsilon����d�=�.       ��W�	�'g�ǩ�A�* 

Average reward per step����D       ��2	k(g�ǩ�A�*

epsilon����,�w*.       ��W�	>Zi�ǩ�A�* 

Average reward per step�����>��       ��2	[i�ǩ�A�*

epsilon������3�.       ��W�	�{k�ǩ�A�* 

Average reward per step�����٦�       ��2	�|k�ǩ�A�*

epsilon����ډ�.       ��W�	v�m�ǩ�A�* 

Average reward per step������J       ��2	H�m�ǩ�A�*

epsilon�����X�.       ��W�	d�o�ǩ�A�* 

Average reward per step������(v       ��2	�o�ǩ�A�*

epsilon�������
.       ��W�	!�q�ǩ�A�* 

Average reward per step�������$       ��2	��q�ǩ�A�*

epsilon����f�tD.       ��W�	�4s�ǩ�A�* 

Average reward per step�����rS        ��2	L5s�ǩ�A�*

epsilon����m���.       ��W�	�t�ǩ�A�* 

Average reward per step�����IF       ��2	��t�ǩ�A�*

epsilon����E�.       ��W�	#�v�ǩ�A�* 

Average reward per step����Vz=u       ��2	��v�ǩ�A�*

epsilon����G��@.       ��W�	^�x�ǩ�A�* 

Average reward per step�����?�       ��2	^�x�ǩ�A�*

epsilon�����%�.       ��W�	�z�ǩ�A�* 

Average reward per step����(�J�       ��2	Ӈz�ǩ�A�*

epsilon�����d>.       ��W�	(�|�ǩ�A�* 

Average reward per step������       ��2	��|�ǩ�A�*

epsilon����B�_P.       ��W�	C�~�ǩ�A�* 

Average reward per step����q�       ��2	 �~�ǩ�A�*

epsilon�������.       ��W�	����ǩ�A�* 

Average reward per step������vy       ��2	����ǩ�A�*

epsilon�������.       ��W�	����ǩ�A�* 

Average reward per step��������       ��2	m���ǩ�A�*

epsilon�����6/�.       ��W�	���ǩ�A�* 

Average reward per step����:#��       ��2	o��ǩ�A�*

epsilon������{.       ��W�	�B��ǩ�A�* 

Average reward per step�����P�       ��2	WC��ǩ�A�*

epsilon����<s�.       ��W�	�t��ǩ�A�* 

Average reward per step����{��       ��2	uu��ǩ�A�*

epsilon����?��.       ��W�	&ߊ�ǩ�A�* 

Average reward per step�������       ��2	�ߊ�ǩ�A�*

epsilon������-e.       ��W�	���ǩ�A�* 

Average reward per step�������       ��2	���ǩ�A�*

epsilon�����_-\.       ��W�	���ǩ�A�* 

Average reward per step����2	�       ��2	R��ǩ�A�*

epsilon��������.       ��W�	f���ǩ�A�* 

Average reward per step�����`	       ��2	����ǩ�A�*

epsilon����G?t.       ��W�	g+��ǩ�A�* 

Average reward per step������̓       ��2	,��ǩ�A�*

epsilon������9.       ��W�	 7��ǩ�A�* 

Average reward per step������K�       ��2	�7��ǩ�A�*

epsilon����BBny.       ��W�	�d��ǩ�A�* 

Average reward per step����<��       ��2	se��ǩ�A�*

epsilon������.       ��W�	����ǩ�A�* 

Average reward per step����8�hG       ��2	צ��ǩ�A�*

epsilon����N���0       ���_	���ǩ�AN*#
!
Average reward per episode�Q߿�
g.       ��W�	=��ǩ�AN*!

total reward per episode  �¥�:�.       ��W�	Yn��ǩ�A�* 

Average reward per step�Q߿�L       ��2	"o��ǩ�A�*

epsilon�Q߿#h�b.       ��W�	����ǩ�A�* 

Average reward per step�Q߿�K-�       ��2	n���ǩ�A�*

epsilon�Q߿�	�.       ��W�	��ǩ�A�* 

Average reward per step�Q߿����       ��2	����ǩ�A�*

epsilon�Q߿܃�.       ��W�	=ѣ�ǩ�A�* 

Average reward per step�Q߿�o>&       ��2	9ң�ǩ�A�*

epsilon�Q߿PTN�.       ��W�	����ǩ�A�* 

Average reward per step�Q߿NL��       ��2	o���ǩ�A�*

epsilon�Q߿x�;%.       ��W�	�0��ǩ�A�* 

Average reward per step�Q߿���       ��2	H1��ǩ�A�*

epsilon�Q߿���y.       ��W�	�ܩ�ǩ�A�* 

Average reward per step�Q߿�IDV       ��2	�ݩ�ǩ�A�*

epsilon�Q߿���3.       ��W�	���ǩ�A�* 

Average reward per step�Q߿���n       ��2	���ǩ�A�*

epsilon�Q߿�H.       ��W�	B#��ǩ�A�* 

Average reward per step�Q߿���       ��2	$��ǩ�A�*

epsilon�Q߿SUp�.       ��W�	�>��ǩ�A�* 

Average reward per step�Q߿�fB�       ��2	�?��ǩ�A�*

epsilon�Q߿sU.       ��W�	(���ǩ�A�* 

Average reward per step�Q߿isr       ��2	���ǩ�A�*

epsilon�Q߿z�.M.       ��W�	���ǩ�A�* 

Average reward per step�Q߿�q��       ��2	���ǩ�A�*

epsilon�Q߿��6^.       ��W�	s,��ǩ�A�* 

Average reward per step�Q߿���(       ��2	0-��ǩ�A�*

epsilon�Q߿��5.       ��W�	=��ǩ�A�* 

Average reward per step�Q߿�U�       ��2	�=��ǩ�A�*

epsilon�Q߿��b�.       ��W�	�Q��ǩ�A�* 

Average reward per step�Q߿���x       ��2	~R��ǩ�A�*

epsilon�Q߿�)vE.       ��W�	Va��ǩ�A�* 

Average reward per step�Q߿K9<I       ��2	,b��ǩ�A�*

epsilon�Q߿���.       ��W�	:z��ǩ�A�* 

Average reward per step�Q߿��I       ��2	�z��ǩ�A�*

epsilon�Q߿R�0U.       ��W�	C���ǩ�A�* 

Average reward per step�Q߿��       ��2	���ǩ�A�*

epsilon�Q߿(��	.       ��W�	����ǩ�A�* 

Average reward per step�Q߿��@$       ��2	j���ǩ�A�*

epsilon�Q߿܅P�.       ��W�	\��ǩ�A�* 

Average reward per step�Q߿����       ��2	&��ǩ�A�*

epsilon�Q߿�G�.       ��W�	ɪ��ǩ�A�* 

Average reward per step�Q߿-�a        ��2	q���ǩ�A�*

epsilon�Q߿JW-.       ��W�	z���ǩ�A�* 

Average reward per step�Q߿ �       ��2	&���ǩ�A�*

epsilon�Q߿vR)�.       ��W�	���ǩ�A�* 

Average reward per step�Q߿WV[�       ��2	,��ǩ�A�*

epsilon�Q߿�,.       ��W�	����ǩ�A�* 

Average reward per step�Q߿kq       ��2	/���ǩ�A�*

epsilon�Q߿`u�U.       ��W�	����ǩ�A�* 

Average reward per step�Q߿tcX�       ��2	����ǩ�A�*

epsilon�Q߿��n�.       ��W�	Z���ǩ�A�* 

Average reward per step�Q߿�AE�       ��2	����ǩ�A�*

epsilon�Q߿n���.       ��W�	���ǩ�A�* 

Average reward per step�Q߿,E˸       ��2	'��ǩ�A�*

epsilon�Q߿�D�.       ��W�	Y0��ǩ�A�* 

Average reward per step�Q߿$���       ��2	1��ǩ�A�*

epsilon�Q߿��.       ��W�	6��ǩ�A�* 

Average reward per step�Q߿�,�       ��2	&7��ǩ�A�*

epsilon�Q߿p���.       ��W�	�[��ǩ�A�* 

Average reward per step�Q߿�d��       ��2	�\��ǩ�A�*

epsilon�Q߿kȥ.       ��W�	����ǩ�A�* 

Average reward per step�Q߿�0�-       ��2	����ǩ�A�*

epsilon�Q߿l�5.       ��W�	����ǩ�A�* 

Average reward per step�Q߿�i�r       ��2	����ǩ�A�*

epsilon�Q߿_�~�.       ��W�	����ǩ�A�* 

Average reward per step�Q߿����       ��2	����ǩ�A�*

epsilon�Q߿�T�|.       ��W�	i���ǩ�A�* 

Average reward per step�Q߿���       ��2	;���ǩ�A�*

epsilon�Q߿J׊.       ��W�	%Y��ǩ�A�* 

Average reward per step�Q߿��]�       ��2	�Y��ǩ�A�*

epsilon�Q߿�x�3.       ��W�	p���ǩ�A�* 

Average reward per step�Q߿u`�M       ��2	���ǩ�A�*

epsilon�Q߿��۔.       ��W�	P��ǩ�A�* 

Average reward per step�Q߿�e�       ��2	���ǩ�A�*

epsilon�Q߿�:/.       ��W�	9���ǩ�A�* 

Average reward per step�Q߿E~,�       ��2	���ǩ�A�*

epsilon�Q߿�p.       ��W�	#���ǩ�A�* 

Average reward per step�Q߿��       ��2	����ǩ�A�*

epsilon�Q߿���}0       ���_	j���ǩ�AO*#
!
Average reward per episode��-�e��}.       ��W�	+���ǩ�AO*!

total reward per episode  ���(�.       ��W�	�(��ǩ�A�* 

Average reward per step��-��凎       ��2	�)��ǩ�A�*

epsilon��-�]h�.       ��W�	_��ǩ�A�* 

Average reward per step��-��>O�       ��2	�_��ǩ�A�*

epsilon��-��b,�.       ��W�	5���ǩ�A�* 

Average reward per step��-���
       ��2	ݚ��ǩ�A�*

epsilon��-��� k.       ��W�	s��ǩ�A�* 

Average reward per step��-�k"�       ��2	*t��ǩ�A�*

epsilon��-��� .       ��W�	���ǩ�A�* 

Average reward per step��-���D�       ��2	����ǩ�A�*

epsilon��-��)��.       ��W�	.Y��ǩ�A�* 

Average reward per step��-�ׯ�5       ��2	�Y��ǩ�A�*

epsilon��-�O���.       ��W�	����ǩ�A�* 

Average reward per step��-�m�       ��2	���ǩ�A�*

epsilon��-�5�j.       ��W�	ڬ��ǩ�A�* 

Average reward per step��-�	6�l       ��2	����ǩ�A�*

epsilon��-�7��.       ��W�	���ǩ�A�* 

Average reward per step��-��x�       ��2	��ǩ�A�*

epsilon��-�rە
.       ��W�	�ǩ�A�* 

Average reward per step��-�&|       ��2	��ǩ�A�*

epsilon��-�[�R.       ��W�	��ǩ�A�* 

Average reward per step��-��       ��2	$��ǩ�A�*

epsilon��-��3�.       ��W�	Y��ǩ�A�* 

Average reward per step��-�Z���       ��2	���ǩ�A�*

epsilon��-��[t�.       ��W�	��
�ǩ�A�* 

Average reward per step��-���/       ��2	Ύ
�ǩ�A�*

epsilon��-�	~�v.       ��W�	�E�ǩ�A�* 

Average reward per step��-���ݩ       ��2	�F�ǩ�A�*

epsilon��-��[�^.       ��W�	q��ǩ�A�* 

Average reward per step��-�l��Y       ��2	:��ǩ�A�*

epsilon��-�۩�v.       ��W�	�&�ǩ�A�* 

Average reward per step��-�g��       ��2	(�ǩ�A�*

epsilon��-��8o�.       ��W�	,��ǩ�A�* 

Average reward per step��-�6l       ��2	˼�ǩ�A�*

epsilon��-��UJ0       ���_	���ǩ�AP*#
!
Average reward per episode�zV��.       ��W�	���ǩ�AP*!

total reward per episode  (�&��.       ��W�	c��ǩ�A�* 

Average reward per step�MUB       ��2	���ǩ�A�*

epsilon�橹.       ��W�	?p�ǩ�A�* 

Average reward per step���       ��2	�p�ǩ�A�*

epsilon����.       ��W�	� �ǩ�A�* 

Average reward per step���e�       ��2	�� �ǩ�A�*

epsilon�����.       ��W�	6�"�ǩ�A�* 

Average reward per step���-       ��2	�"�ǩ�A�*

epsilon��af.       ��W�	�%�ǩ�A�* 

Average reward per step����       ��2	Q%�ǩ�A�*

epsilon�r�:�.       ��W�	$E'�ǩ�A�* 

Average reward per step�Q?H�       ��2	�E'�ǩ�A�*

epsilon�QEy�.       ��W�	%u)�ǩ�A�* 

Average reward per step��_ۖ       ��2	�u)�ǩ�A�*

epsilon���L.       ��W�	�5+�ǩ�A�* 

Average reward per step�#;�?       ��2	i6+�ǩ�A�*

epsilon���N.       ��W�	�o.�ǩ�A�* 

Average reward per step�[ ��       ��2	�p.�ǩ�A�*

epsilon�T�ǉ.       ��W�	p1�ǩ�A�* 

Average reward per step��>4       ��2	�p1�ǩ�A�*

epsilon� K��.       ��W�	-�3�ǩ�A�* 

Average reward per step�����       ��2	��3�ǩ�A�*

epsilon�h�<+.       ��W�	u5�ǩ�A�* 

Average reward per step����       ��2	�u5�ǩ�A�*

epsilon�����.       ��W�	'8�ǩ�A�* 

Average reward per step��h��       ��2	�8�ǩ�A�*

epsilon�+\G.       ��W�	��9�ǩ�A�* 

Average reward per step��:to       ��2	i�9�ǩ�A�*

epsilon��TNZ.       ��W�	{�;�ǩ�A�* 

Average reward per step�b"�7       ��2	^�;�ǩ�A�*

epsilon�/q
�.       ��W�	��=�ǩ�A�* 

Average reward per step��D7       ��2	'�=�ǩ�A�*

epsilon���#�.       ��W�	��?�ǩ�A�* 

Average reward per step�H@�       ��2	��?�ǩ�A�*

epsilon�A��:.       ��W�	q<B�ǩ�A�* 

Average reward per step�2%�|       ��2	=B�ǩ�A�*

epsilon��!�L0       ���_	�YB�ǩ�AQ*#
!
Average reward per episode�q�5��.       ��W�	`ZB�ǩ�AQ*!

total reward per episode  'ú�6.       ��W�	(�E�ǩ�A�* 

Average reward per step�q�MD�r       ��2	��E�ǩ�A�*

epsilon�q�Ar �.       ��W�	zH�ǩ�A�* 

Average reward per step�q���_       ��2	H�ǩ�A�*

epsilon�q��ؿ.       ��W�	�?J�ǩ�A�* 

Average reward per step�q�f�Z%       ��2	S@J�ǩ�A�*

epsilon�q�X��e.       ��W�	KWL�ǩ�A�* 

Average reward per step�q�MD       ��2	XL�ǩ�A�*

epsilon�q���.       ��W�	��M�ǩ�A�* 

Average reward per step�q�'� �       ��2	�M�ǩ�A�*

epsilon�q�����.       ��W�	 P�ǩ�A�* 

Average reward per step�q�$�Q       ��2	� P�ǩ�A�*

epsilon�q���B|.       ��W�	�R�ǩ�A�* 

Average reward per step�q�dk]�       ��2	hR�ǩ�A�*

epsilon�q���sS.       ��W�	6"T�ǩ�A�* 

Average reward per step�q���r       ��2	1#T�ǩ�A�*

epsilon�q����%.       ��W�	P7V�ǩ�A�* 

Average reward per step�q�Fh�       ��2	8V�ǩ�A�*

epsilon�q��*�.       ��W�	�VX�ǩ�A�* 

Average reward per step�q��6�       ��2	�WX�ǩ�A�*

epsilon�q�~'2�.       ��W�	�Z�ǩ�A�* 

Average reward per step�q���wf       ��2	�Z�ǩ�A�*

epsilon�q���#�.       ��W�	^�\�ǩ�A�* 

Average reward per step�q����       ��2	�\�ǩ�A�*

epsilon�q����t.       ��W�	<�^�ǩ�A�* 

Average reward per step�q����       ��2	��^�ǩ�A�*

epsilon�q��p�!.       ��W�	�]`�ǩ�A�* 

Average reward per step�q�J�x       ��2	h^`�ǩ�A�*

epsilon�q�c�q.       ��W�	4,c�ǩ�A�* 

Average reward per step�q���7U       ��2	-c�ǩ�A�*

epsilon�q�����.       ��W�	Ze�ǩ�A�* 

Average reward per step�q�א�       ��2	�Ze�ǩ�A�*

epsilon�q�g��.       ��W�	��g�ǩ�A�* 

Average reward per step�q��/W       ��2	��g�ǩ�A�*

epsilon�q�����.       ��W�	��i�ǩ�A�* 

Average reward per step�q���       ��2	9�i�ǩ�A�*

epsilon�q�{Ikj.       ��W�	O<m�ǩ�A�* 

Average reward per step�q����e       ��2	.=m�ǩ�A�*

epsilon�q����.       ��W�	�oo�ǩ�A�* 

Average reward per step�q�H	2       ��2	�po�ǩ�A�*

epsilon�q���(F.       ��W�	L�q�ǩ�A�* 

Average reward per step�q��	        ��2	ߊq�ǩ�A�*

epsilon�q�;x�.       ��W�	/�s�ǩ�A�* 

Average reward per step�q�x�1�       ��2	ۥs�ǩ�A�*

epsilon�q�J!��.       ��W�	�u�ǩ�A�* 

Average reward per step�q����#       ��2	eu�ǩ�A�*

epsilon�q���E�.       ��W�	�Iw�ǩ�A�* 

Average reward per step�q��z&-       ��2	0Jw�ǩ�A�*

epsilon�q�t�d�.       ��W�	jmy�ǩ�A�* 

Average reward per step�q���;       ��2	3ny�ǩ�A�*

epsilon�q���O�.       ��W�	j�{�ǩ�A�* 

Average reward per step�q�����       ��2	I�{�ǩ�A�*

epsilon�q��˙R.       ��W�	�a}�ǩ�A�* 

Average reward per step�q��	�       ��2	�b}�ǩ�A�*

epsilon�q�|ֈ�.       ��W�	3��ǩ�A�* 

Average reward per step�q�7sA       ��2	֋�ǩ�A�*

epsilon�q��A�.       ��W�	���ǩ�A�* 

Average reward per step�q�|��       ��2	й��ǩ�A�*

epsilon�q�\T|�.       ��W�	�̓�ǩ�A�* 

Average reward per step�q�+�D       ��2	-΃�ǩ�A�*

epsilon�q�Eճ.       ��W�	��ǩ�A�* 

Average reward per step�q�j��       ��2	���ǩ�A�*

epsilon�q���@.       ��W�	���ǩ�A�* 

Average reward per step�q�p�       ��2	y ��ǩ�A�*

epsilon�q�b)�.       ��W�	U���ǩ�A�* 

Average reward per step�q�L��l       ��2	콉�ǩ�A�*

epsilon�q�2���.       ��W�	����ǩ�A�* 

Average reward per step�q�;
?�       ��2	b���ǩ�A�*

epsilon�q��^@�.       ��W�	�)��ǩ�A�* 

Average reward per step�q�&��       ��2	�*��ǩ�A�*

epsilon�q�CR�.       ��W�	"Q��ǩ�A�* 

Average reward per step�q�n�X       ��2	�Q��ǩ�A�*

epsilon�q���a.       ��W�	U��ǩ�A�* 

Average reward per step�q�t��       ��2	b��ǩ�A�*

epsilon�q�i��0.       ��W�	r7��ǩ�A�* 

Average reward per step�q��v �       ��2	28��ǩ�A�*

epsilon�q���S�.       ��W�	O[��ǩ�A�* 

Average reward per step�q�c_��       ��2	�[��ǩ�A�*

epsilon�q��W�.       ��W�	'ٗ�ǩ�A�* 

Average reward per step�q���l       ��2	�ٗ�ǩ�A�*

epsilon�q�dY
.       ��W�	�	��ǩ�A�* 

Average reward per step�q����>       ��2	J
��ǩ�A�*

epsilon�q���-.       ��W�	���ǩ�A�* 

Average reward per step�q�9��       ��2	O��ǩ�A�*

epsilon�q�"�	.       ��W�	n���ǩ�A�* 

Average reward per step�q�^��       ��2		���ǩ�A�*

epsilon�q�����.       ��W�	����ǩ�A�* 

Average reward per step�q�+��g       ��2	8���ǩ�A�*

epsilon�q�l>��.       ��W�	���ǩ�A�* 

Average reward per step�q�G=ON       ��2	���ǩ�A�*

epsilon�q�)x]�.       ��W�	2��ǩ�A�* 

Average reward per step�q��4       ��2	Y3��ǩ�A�*

epsilon�q���mB.       ��W�	o���ǩ�A�* 

Average reward per step�q�����       ��2	����ǩ�A�*

epsilon�q��0K.       ��W�	�̨�ǩ�A�* 

Average reward per step�q�� ti       ��2	lͨ�ǩ�A�*

epsilon�q�נ�	.       ��W�	���ǩ�A�* 

Average reward per step�q�^)=T       ��2	���ǩ�A�*

epsilon�q�z�{.       ��W�	:��ǩ�A�* 

Average reward per step�q��0�h       ��2	���ǩ�A�*

epsilon�q��E�.       ��W�	5���ǩ�A�* 

Average reward per step�q��v7S       ��2	���ǩ�A�*

epsilon�q��i<h.       ��W�	>��ǩ�A�* 

Average reward per step�q�5D�       ��2	���ǩ�A�*

epsilon�q��К.       ��W�	Ū��ǩ�A�* 

Average reward per step�q��x        ��2	뫳�ǩ�A�*

epsilon�q�H �Z.       ��W�	&Q��ǩ�A�* 

Average reward per step�q�t�       ��2	�Q��ǩ�A�*

epsilon�q��_E=0       ���_	�n��ǩ�AR*#
!
Average reward per episode�О��V�_.       ��W�	;o��ǩ�AR*!

total reward per episode  ����R.       ��W�	���ǩ�A�* 

Average reward per step�О��Z�       ��2	���ǩ�A�*

epsilon�О���1a.       ��W�	,Խ�ǩ�A�* 

Average reward per step�О�W`M�       ��2	�Խ�ǩ�A�*

epsilon�О�l#�.       ��W�	���ǩ�A�* 

Average reward per step�О��(p$       ��2	o��ǩ�A�*

epsilon�О�E��k.       ��W�	C��ǩ�A�* 

Average reward per step�О�Ns	X       ��2	���ǩ�A�*

epsilon�О�V+7z.       ��W�	<���ǩ�A�* 

Average reward per step�О���i       ��2	���ǩ�A�*

epsilon�О��.       ��W�	����ǩ�A�* 

Average reward per step�О��=�       ��2	����ǩ�A�*

epsilon�О�W8��.       ��W�	
���ǩ�A�* 

Average reward per step�О����       ��2	����ǩ�A�*

epsilon�О����.       ��W�	m���ǩ�A�* 

Average reward per step�О��e��       ��2	 ���ǩ�A�*

epsilon�О�O���.       ��W�	���ǩ�A�* 

Average reward per step�О����       ��2	&��ǩ�A�*

epsilon�О��w.       ��W�	+2��ǩ�A�* 

Average reward per step�О�}c�}       ��2	�2��ǩ�A�*

epsilon�О�4k}�.       ��W�	�a��ǩ�A�* 

Average reward per step�О��Df       ��2	cb��ǩ�A�*

epsilon�О��4{.       ��W�	O���ǩ�A�* 

Average reward per step�О��UÎ       ��2	-���ǩ�A�*

epsilon�О�I�gI.       ��W�	�x��ǩ�A�* 

Average reward per step�О�	l�       ��2	�y��ǩ�A�*

epsilon�О��j�.       ��W�	����ǩ�A�* 

Average reward per step�О��̭�       ��2	����ǩ�A�*

epsilon�О�&��.       ��W�	¤��ǩ�A�* 

Average reward per step�О�����       ��2	����ǩ�A�*

epsilon�О�T���.       ��W�	@���ǩ�A�* 

Average reward per step�О�D�"U       ��2	����ǩ�A�*

epsilon�О�W~.       ��W�	����ǩ�A�* 

Average reward per step�О�ݝ>O       ��2	����ǩ�A�*

epsilon�О�s\.       ��W�	�Y��ǩ�A�* 

Average reward per step�О�E_,d       ��2	\Z��ǩ�A�*

epsilon�О�\K �.       ��W�	�2��ǩ�A�* 

Average reward per step�О��j3�       ��2	4��ǩ�A�*

epsilon�О�F?Ě.       ��W�	����ǩ�A�* 

Average reward per step�О�p���       ��2	 ���ǩ�A�*

epsilon�О��$�C.       ��W�	����ǩ�A�* 

Average reward per step�О�Z9�'       ��2	Q���ǩ�A�*

epsilon�О�P�.       ��W�	i��ǩ�A�* 

Average reward per step�О���4       ��2	~��ǩ�A�*

epsilon�О��oX�.       ��W�	�2��ǩ�A�* 

Average reward per step�О�sF�       ��2	Y3��ǩ�A�*

epsilon�О�s�x.       ��W�	Â��ǩ�A�* 

Average reward per step�О�hKf       ��2	V���ǩ�A�*

epsilon�О���
B.       ��W�	���ǩ�A�* 

Average reward per step�О��[�       ��2	v��ǩ�A�*

epsilon�О�o&|4.       ��W�	�:��ǩ�A�* 

Average reward per step�О�>p�3       ��2	O;��ǩ�A�*

epsilon�О�GP3�.       ��W�	@K��ǩ�A�* 

Average reward per step�О��7�       ��2	L��ǩ�A�*

epsilon�О�)�4.       ��W�	{���ǩ�A�* 

Average reward per step�О�U���       ��2	���ǩ�A�*

epsilon�О�?�׌.       ��W�	����ǩ�A�* 

Average reward per step�О��T�Y       ��2	���ǩ�A�*

epsilon�О�44.       ��W�	�P��ǩ�A�* 

Average reward per step�О�B�3�       ��2	�Q��ǩ�A�*

epsilon�О���IV.       ��W�	�^��ǩ�A�* 

Average reward per step�О���G       ��2	W_��ǩ�A�*

epsilon�О�Q޶.       ��W�	+m��ǩ�A�* 

Average reward per step�О��o��       ��2	�m��ǩ�A�*

epsilon�О� �l.       ��W�	���ǩ�A�* 

Average reward per step�О�P��       ��2	����ǩ�A�*

epsilon�О��q;i.       ��W�	���ǩ�A�* 

Average reward per step�О���9�       ��2	���ǩ�A�*

epsilon�О����O0       ���_	 ��ǩ�AS*#
!
Average reward per episode��g��Ln.       ��W�	���ǩ�AS*!

total reward per episode  ���y�.       ��W�	T��ǩ�A�* 

Average reward per step��g����E       ��2	��ǩ�A�*

epsilon��g�(P^�.       ��W�	v��ǩ�A�* 

Average reward per step��g�vW       ��2	v��ǩ�A�*

epsilon��g�q���.       ��W�	��
�ǩ�A�* 

Average reward per step��g�-���       ��2	�
�ǩ�A�*

epsilon��g�n���.       ��W�	�ǩ�A�* 

Average reward per step��g����       ��2	@�ǩ�A�*

epsilon��g�3���.       ��W�	���ǩ�A�* 

Average reward per step��g�Y)�W       ��2	Á�ǩ�A�*

epsilon��g��U��.       ��W�	���ǩ�A�* 

Average reward per step��g��=.       ��2	���ǩ�A�*

epsilon��g�]��.       ��W�	ZF�ǩ�A�* 

Average reward per step��g�p[�       ��2	AG�ǩ�A�*

epsilon��g��o.       ��W�	���ǩ�A�* 

Average reward per step��g���       ��2	[��ǩ�A�*

epsilon��g�^!�.       ��W�	���ǩ�A�* 

Average reward per step��g�%��       ��2	q��ǩ�A�*

epsilon��g��g�.       ��W�	S%�ǩ�A�* 

Average reward per step��g��6ie       ��2	F&�ǩ�A�*

epsilon��g��e�.       ��W�	JE�ǩ�A�* 

Average reward per step��g�Q�k�       ��2	�E�ǩ�A�*

epsilon��g�"��].       ��W�	�X�ǩ�A�* 

Average reward per step��g��5�U       ��2	CY�ǩ�A�*

epsilon��g��sd�.       ��W�	�h�ǩ�A�* 

Average reward per step��g��o�_       ��2	�i�ǩ�A�*

epsilon��g��Z�_.       ��W�	2� �ǩ�A�* 

Average reward per step��g�����       ��2	*� �ǩ�A�*

epsilon��g�~�_.       ��W�	�@"�ǩ�A�* 

Average reward per step��g��İm       ��2	�A"�ǩ�A�*

epsilon��g���Eh.       ��W�	�c$�ǩ�A�* 

Average reward per step��g�}��       ��2	�d$�ǩ�A�*

epsilon��g����.       ��W�	N�&�ǩ�A�* 

Average reward per step��g�9Ds       ��2	�&�ǩ�A�*

epsilon��g�8�1�.       ��W�	��(�ǩ�A�* 

Average reward per step��g�R"�       ��2	��(�ǩ�A�*

epsilon��g�?���.       ��W�	��*�ǩ�A�* 

Average reward per step��g��|X�       ��2	7�*�ǩ�A�*

epsilon��g�{-|v.       ��W�	�T,�ǩ�A�* 

Average reward per step��g�{Y��       ��2	�U,�ǩ�A�*

epsilon��g��N.       ��W�	vk.�ǩ�A�* 

Average reward per step��g��t�       ��2	Ql.�ǩ�A�*

epsilon��g�y�o�0       ���_	��.�ǩ�AT*#
!
Average reward per episode۶����F.       ��W�	D�.�ǩ�AT*!

total reward per episode  �֮'�.       ��W�	�3�ǩ�A�* 

Average reward per step۶���Xrx       ��2	� 3�ǩ�A�*

epsilon۶���t_".       ��W�	B�4�ǩ�A�* 

Average reward per step۶���Ur       ��2	�4�ǩ�A�*

epsilon۶��N�-�.       ��W�	��6�ǩ�A�* 

Average reward per step۶�� ���       ��2	=�6�ǩ�A�*

epsilon۶��Ft>.       ��W�	T 9�ǩ�A�* 

Average reward per step۶��<li�       ��2	9�ǩ�A�*

epsilon۶���w��.       ��W�	�2;�ǩ�A�* 

Average reward per step۶��� w�       ��2	{3;�ǩ�A�*

epsilon۶����.       ��W�	ir=�ǩ�A�* 

Average reward per step۶���       ��2	.s=�ǩ�A�*

epsilon۶��'+~.       ��W�	��>�ǩ�A�* 

Average reward per step۶��΋�       ��2	;�>�ǩ�A�*

epsilon۶��u��Q.       ��W�	^�A�ǩ�A�* 

Average reward per step۶����4       ��2	�A�ǩ�A�*

epsilon۶��cC�.       ��W�	�B�ǩ�A�* 

Average reward per step۶��껜&       ��2	��B�ǩ�A�*

epsilon۶���]L�.       ��W�	�E�ǩ�A�* 

Average reward per step۶��U��       ��2	�E�ǩ�A�*

epsilon۶�����.       ��W�	�6G�ǩ�A�* 

Average reward per step۶��s��       ��2	�7G�ǩ�A�*

epsilon۶��%h.       ��W�	� I�ǩ�A�* 

Average reward per step۶��}�       ��2	\I�ǩ�A�*

epsilon۶��t�!.       ��W�	&K�ǩ�A�* 

Average reward per step۶��E�)       ��2	�&K�ǩ�A�*

epsilon۶����.       ��W�	GM�ǩ�A�* 

Average reward per step۶��Y�ӛ       ��2	�GM�ǩ�A�*

epsilon۶����.       ��W�	�`O�ǩ�A�* 

Average reward per step۶��N�v       ��2	�aO�ǩ�A�*

epsilon۶��"�a.       ��W�	wQ�ǩ�A�* 

Average reward per step۶���f��       ��2	�wQ�ǩ�A�*

epsilon۶��;�l�.       ��W�	Tr��ǩ�A�* 

Average reward per step۶����ZS       ��2	s��ǩ�A�*

epsilon۶���^�.       ��W�	��ǩ�A�* 

Average reward per step۶����(L       ��2	���ǩ�A�*

epsilon۶��@j�`.       ��W�	$��ǩ�A�* 

Average reward per step۶��<]L�       ��2	���ǩ�A�*

epsilon۶���7K.       ��W�	7��ǩ�A�* 

Average reward per step۶��O��       ��2	���ǩ�A�*

epsilon۶�����.       ��W�	����ǩ�A�* 

Average reward per step۶�����K       ��2	8���ǩ�A�*

epsilon۶��<��;.       ��W�	���ǩ�A�* 

Average reward per step۶����       ��2	u��ǩ�A�*

epsilon۶��1�m6.       ��W�	���ǩ�A�* 

Average reward per step۶�����Y       ��2	D��ǩ�A�*

epsilon۶��1�%.       ��W�	͓��ǩ�A�* 

Average reward per step۶����d�       ��2	����ǩ�A�*

epsilon۶���>��.       ��W�	�
��ǩ�A�* 

Average reward per step۶��k��t       ��2	���ǩ�A�*

epsilon۶�����J.       ��W�	y;��ǩ�A�* 

Average reward per step۶���\`       ��2	<��ǩ�A�*

epsilon۶���Ym.       ��W�	l��ǩ�A�* 

Average reward per step۶���*��       ��2	�l��ǩ�A�*

epsilon۶��}��\.       ��W�	���ǩ�A�* 

Average reward per step۶��'�       ��2	���ǩ�A�*

epsilon۶��HQ��0       ���_	%��ǩ�AU*#
!
Average reward per episode�m��I���.       ��W�	���ǩ�AU*!

total reward per episode  ��~Y�.       ��W�	����ǩ�A�* 

Average reward per step�m����S4       ��2	:���ǩ�A�*

epsilon�m����.       ��W�	�˲�ǩ�A�* 

Average reward per step�m�����L       ��2	l̲�ǩ�A�*

epsilon�m��U��.       ��W�	�C��ǩ�A�* 

Average reward per step�m��Bՙ�       ��2	D��ǩ�A�*

epsilon�m���:�0.       ��W�	dY��ǩ�A�* 

Average reward per step�m����)       ��2	-Z��ǩ�A�*

epsilon�m����1�.       ��W�	%Y��ǩ�A�* 

Average reward per step�m��G��       ��2	�Y��ǩ�A�*

epsilon�m��x3�.       ��W�	�o��ǩ�A�* 

Average reward per step�m���)Y�       ��2	&p��ǩ�A�*

epsilon�m��]���.       ��W�	k���ǩ�A�* 

Average reward per step�m������       ��2	����ǩ�A�*

epsilon�m����.       ��W�	����ǩ�A�* 

Average reward per step�m���       ��2	g���ǩ�A�*

epsilon�m��|N�.       ��W�	`���ǩ�A�* 

Average reward per step�m����)t       ��2	���ǩ�A�*

epsilon�m��X�b-.       ��W�	}���ǩ�A�* 

Average reward per step�m��޽E=       ��2	p���ǩ�A�*

epsilon�m����sY.       ��W�	����ǩ�A�* 

Average reward per step�m��3�v�       ��2	x���ǩ�A�*

epsilon�m����f�.       ��W�	�
��ǩ�A�* 

Average reward per step�m��R��       ��2	���ǩ�A�*

epsilon�m��Eg�j.       ��W�	
���ǩ�A�* 

Average reward per step�m��IU�       ��2	ۅ��ǩ�A�*

epsilon�m��e�(h.       ��W�	�O��ǩ�A�* 

Average reward per step�m��2F#       ��2	jP��ǩ�A�*

epsilon�m���wm�.       ��W�	A���ǩ�A�* 

Average reward per step�m��E���       ��2	����ǩ�A�*

epsilon�m���"�0.       ��W�	���ǩ�A�* 

Average reward per step�m���Z��       ��2	\��ǩ�A�*

epsilon�m��83��.       ��W�	L4��ǩ�A�* 

Average reward per step�m��lw�       ��2	�4��ǩ�A�*

epsilon�m��=W�<.       ��W�	ca��ǩ�A�* 

Average reward per step�m��Ò        ��2	Jb��ǩ�A�*

epsilon�m�����.       ��W�	����ǩ�A�* 

Average reward per step�m����b�       ��2	U���ǩ�A�*

epsilon�m��|��.       ��W�	���ǩ�A�* 

Average reward per step�m��||�w       ��2	,��ǩ�A�*

epsilon�m��M+}q.       ��W�	�2��ǩ�A�* 

Average reward per step�m��vŷ�       ��2	]3��ǩ�A�*

epsilon�m��f�}�.       ��W�	�N��ǩ�A�* 

Average reward per step�m��1^�       ��2	�O��ǩ�A�*

epsilon�m���̊u.       ��W�	Ho�ǩ�A�* 

Average reward per step�m���f��       ��2	�o�ǩ�A�*

epsilon�m�����.       ��W�	���ǩ�A�* 

Average reward per step�m���7�d       ��2	~��ǩ�A�*

epsilon�m��j	Py.       ��W�	�!�ǩ�A�* 

Average reward per step�m���Ir       ��2	��!�ǩ�A�*

epsilon�m��F8�.       ��W�	J�#�ǩ�A�* 

Average reward per step�m���As�       ��2	��#�ǩ�A�*

epsilon�m��v	�W.       ��W�	��%�ǩ�A�* 

Average reward per step�m���jd�       ��2	z�%�ǩ�A�*

epsilon�m���xZ.       ��W�	��'�ǩ�A�* 

Average reward per step�m��a�Q�       ��2	S�'�ǩ�A�*

epsilon�m��Nj�.       ��W�	]�)�ǩ�A�* 

Average reward per step�m��m��       ��2	��)�ǩ�A�*

epsilon�m��#�S`.       ��W�	-	,�ǩ�A�* 

Average reward per step�m����P�       ��2	�	,�ǩ�A�*

epsilon�m��P\~.       ��W�	��-�ǩ�A�* 

Average reward per step�m���T�       ��2	a�-�ǩ�A�*

epsilon�m��ⴺ%0       ���_	]�-�ǩ�AV*#
!
Average reward per episode�R��(Իd.       ��W�	��-�ǩ�AV*!

total reward per episode  �8�8.       ��W�	�%2�ǩ�A�* 

Average reward per step�R��̕O       ��2	g&2�ǩ�A�*

epsilon�R��8U2�.       ��W�	�@4�ǩ�A�* 

Average reward per step�R����       ��2	�A4�ǩ�A�*

epsilon�R��B4A.       ��W�	?�5�ǩ�A�* 

Average reward per step�R��LF�       ��2	ڬ5�ǩ�A�*

epsilon�R���6��.       ��W�	�&7�ǩ�A�* 

Average reward per step�R���ڭ       ��2	�'7�ǩ�A�*

epsilon�R��#*�.       ��W�	p@9�ǩ�A�* 

Average reward per step�R��)�       ��2	-A9�ǩ�A�*

epsilon�R��[��.       ��W�	�w;�ǩ�A�* 

Average reward per step�R���e��       ��2	�x;�ǩ�A�*

epsilon�R���oS�.       ��W�	$�=�ǩ�A�* 

Average reward per step�R�����       ��2	��=�ǩ�A�*

epsilon�R���B�.       ��W�	�?�ǩ�A�* 

Average reward per step�R���z       ��2	X?�ǩ�A�*

epsilon�R������.       ��W�	�TA�ǩ�A�* 

Average reward per step�R��q��       ��2	�UA�ǩ�A�*

epsilon�R��/}��.       ��W�	cC�ǩ�A�* 

Average reward per step�R����"�       ��2	�cC�ǩ�A�*

epsilon�R���Z�
.       ��W�	�wE�ǩ�A�* 

Average reward per step�R����       ��2	�xE�ǩ�A�*

epsilon�R���.       ��W�	j�G�ǩ�A�* 

Average reward per step�R�����a       ��2	7�G�ǩ�A�*

epsilon�R��O@_;.       ��W�	�I�ǩ�A�* 

Average reward per step�R��S�       ��2	еI�ǩ�A�*

epsilon�R����2v.       ��W�	Q�K�ǩ�A�* 

Average reward per step�R��ю��       ��2	�K�ǩ�A�*

epsilon�R��M��!.       ��W�	irM�ǩ�A�* 

Average reward per step�R���(_       ��2	sM�ǩ�A�*

epsilon�R��(�$�.       ��W�	J�O�ǩ�A�* 

Average reward per step�R���v�       ��2	��O�ǩ�A�*

epsilon�R����~�.       ��W�	e�Q�ǩ�A�* 

Average reward per step�R��y�i�       ��2	T�Q�ǩ�A�*

epsilon�R��Fg.       ��W�	&T�ǩ�A�* 

Average reward per step�R���\g       ��2	�T�ǩ�A�*

epsilon�R���M�0       ���_	&5T�ǩ�AW*#
!
Average reward per episode�	���j(.       ��W�	�5T�ǩ�AW*!

total reward per episode  �+	�.       ��W�	zX�ǩ�A�* 

Average reward per step�	���HT       ��2	�zX�ǩ�A�*

epsilon�	�H��x.       ��W�	��Y�ǩ�A�* 

Average reward per step�	�˩ͨ       ��2	� Z�ǩ�A�*

epsilon�	�sG��.       ��W�	�\�ǩ�A�* 

Average reward per step�	��gm       ��2	� \�ǩ�A�*

epsilon�	��tn.       ��W�	BA^�ǩ�A�* 

Average reward per step�	��Ů�       ��2	B^�ǩ�A�*

epsilon�	��>�.       ��W�	we`�ǩ�A�* 

Average reward per step�	�}���       ��2	Af`�ǩ�A�*

epsilon�	�~y��.       ��W�	�b�ǩ�A�* 

Average reward per step�	�ߦC�       ��2	��b�ǩ�A�*

epsilon�	��w��.       ��W�	��d�ǩ�A�* 

Average reward per step�	�Þ��       ��2	r�d�ǩ�A�*

epsilon�	���.       ��W�	��f�ǩ�A�* 

Average reward per step�	��_�       ��2	��f�ǩ�A�*

epsilon�	���U.       ��W�	�Dh�ǩ�A�* 

Average reward per step�	����       ��2	�Eh�ǩ�A�*

epsilon�	�:�Vm.       ��W�	�_j�ǩ�A�* 

Average reward per step�	�rİ       ��2	�`j�ǩ�A�*

epsilon�	���.       ��W�	b�k�ǩ�A�* 

Average reward per step�	����I       ��2	+�k�ǩ�A�*

epsilon�	��L�e.       ��W�	p#n�ǩ�A�* 

Average reward per step�	�����       ��2	5$n�ǩ�A�*

epsilon�	�Y�-.       ��W�	z�o�ǩ�A�* 

Average reward per step�	��,��       ��2	�o�ǩ�A�*

epsilon�	�P�D.       ��W�	��q�ǩ�A�* 

Average reward per step�	��a       ��2	d�q�ǩ�A�*

epsilon�	��V8�.       ��W�	%�s�ǩ�A�* 

Average reward per step�	��:�       ��2	��s�ǩ�A�*

epsilon�	����[.       ��W�	Iv�ǩ�A�* 

Average reward per step�	��]�E       ��2	�Iv�ǩ�A�*

epsilon�	��Z��.       ��W�	ux�ǩ�A�* 

Average reward per step�	���j       ��2	�ux�ǩ�A�*

epsilon�	�I
�3.       ��W�	��z�ǩ�A�* 

Average reward per step�	��>�o       ��2	��z�ǩ�A�*

epsilon�	��Aj0.       ��W�	�|�ǩ�A�* 

Average reward per step�	���c�       ��2	�|�ǩ�A�*

epsilon�	��K.       ��W�	='~�ǩ�A�* 

Average reward per step�	����O       ��2	(~�ǩ�A�*

epsilon�	�ay�.       ��W�	�|��ǩ�A�* 

Average reward per step�	�K�^9       ��2	F}��ǩ�A�*

epsilon�	��DY.       ��W�	͔��ǩ�A�* 

Average reward per step�	��-*&       ��2	d���ǩ�A�*

epsilon�	���-.       ��W�	⬄�ǩ�A�* 

Average reward per step�	��Y�       ��2	����ǩ�A�*

epsilon�	�}̫�.       ��W�	M��ǩ�A�* 

Average reward per step�	���1�       ��2	'��ǩ�A�*

epsilon�	��b�.       ��W�	�i��ǩ�A�* 

Average reward per step�	��>�       ��2		k��ǩ�A�*

epsilon�	��e��.       ��W�	1���ǩ�A�* 

Average reward per step�	�-��       ��2	ė��ǩ�A�*

epsilon�	�n@.       ��W�	����ǩ�A�* 

Average reward per step�	��
��       ��2	����ǩ�A�*

epsilon�	�b�A�.       ��W�	.Ǝ�ǩ�A�* 

Average reward per step�	��x�E       ��2	�Ǝ�ǩ�A�*

epsilon�	��J�.       ��W�	���ǩ�A�* 

Average reward per step�	�4�js       ��2	���ǩ�A�*

epsilon�	�Al�.       ��W�	%"��ǩ�A�* 

Average reward per step�	����H       ��2	#��ǩ�A�*

epsilon�	��!s.       ��W�	���ǩ�A�* 

Average reward per step�	��c��       ��2	G��ǩ�A�*

epsilon�	�$��>.       ��W�	4��ǩ�A�* 

Average reward per step�	�Rf�       ��2	���ǩ�A�*

epsilon�	���*.       ��W�	U��ǩ�A�* 

Average reward per step�	��K�       ��2	�U��ǩ�A�*

epsilon�	�-��.       ��W�	cd��ǩ�A�* 

Average reward per step�	�;�1i       ��2	�d��ǩ�A�*

epsilon�	�zUy�.       ��W�	o��ǩ�A�* 

Average reward per step�	��53�       ��2	*p��ǩ�A�*

epsilon�	��J�S.       ��W�	ߌ��ǩ�A�* 

Average reward per step�	����       ��2	m���ǩ�A�*

epsilon�	�TS;40       ���_	����ǩ�AX*#
!
Average reward per episode  P�ڱK�.       ��W�	j���ǩ�AX*!

total reward per episode  ��,]�	.       ��W�	4e��ǩ�A�* 

Average reward per step  P�nD'�       ��2	�e��ǩ�A�*

epsilon  P��%@.       ��W�	*s��ǩ�A�* 

Average reward per step  P�)�W       ��2	�s��ǩ�A�*

epsilon  P�لH�.       ��W�	߈��ǩ�A�* 

Average reward per step  P�R|'       ��2	����ǩ�A�*

epsilon  P��8׌.       ��W�	0���ǩ�A�* 

Average reward per step  P�d=��       ��2	ǟ��ǩ�A�*

epsilon  P��9F).       ��W�	=ԭ�ǩ�A�* 

Average reward per step  P�u!��       ��2	�ԭ�ǩ�A�*

epsilon  P��l#�.       ��W�	 U��ǩ�A�* 

Average reward per step  P�a�       ��2	�U��ǩ�A�*

epsilon  P��շ6.       ��W�	���ǩ�A�* 

Average reward per step  P��n�       ��2	����ǩ�A�*

epsilon  P���y.       ��W�	mʳ�ǩ�A�* 

Average reward per step  P�B��       ��2	6˳�ǩ�A�*

epsilon  P�CR�=.       ��W�	����ǩ�A�* 

Average reward per step  P�!�Qz       ��2	[���ǩ�A�*

epsilon  P�3�#..       ��W�	�Ϸ�ǩ�A�* 

Average reward per step  P���t       ��2	Jз�ǩ�A�*

epsilon  P��5�.       ��W�	�	��ǩ�A�* 

Average reward per step  P�x�07       ��2	l
��ǩ�A�*

epsilon  P�c��.       ��W�	�#��ǩ�A�* 

Average reward per step  P�&F�G       ��2	h$��ǩ�A�*

epsilon  P�u��.       ��W�	JF��ǩ�A�* 

Average reward per step  P�3��N       ��2	G��ǩ�A�*

epsilon  P��� +.       ��W�	����ǩ�A�* 

Average reward per step  P���       ��2	,���ǩ�A�*

epsilon  P���b.       ��W�	 ��ǩ�A�* 

Average reward per step  P��W;       ��2	���ǩ�A�*

epsilon  P��*3�.       ��W�	�K��ǩ�A�* 

Average reward per step  P��	+�       ��2	�L��ǩ�A�*

epsilon  P����S.       ��W�	8i��ǩ�A�* 

Average reward per step  P�QM-D       ��2	+j��ǩ�A�*

epsilon  P�þ��.       ��W�	3��ǩ�A�* 

Average reward per step  P��G�       ��2	�3��ǩ�A�*

epsilon  P��W�.       ��W�	 8��ǩ�A�* 

Average reward per step  P�a?�~       ��2	�8��ǩ�A�*

epsilon  P�����.       ��W�	�}��ǩ�A�* 

Average reward per step  P��i)�       ��2	�~��ǩ�A�*

epsilon  P�z��G.       ��W�	����ǩ�A�* 

Average reward per step  P��V�J       ��2	����ǩ�A�*

epsilon  P�؛�.       ��W�	���ǩ�A�* 

Average reward per step  P�c�H�       ��2	���ǩ�A�*

epsilon  P�(8�D.       ��W�	�|��ǩ�A�* 

Average reward per step  P����1       ��2	�}��ǩ�A�*

epsilon  P�B��..       ��W�	0���ǩ�A�* 

Average reward per step  P�ES�       ��2	���ǩ�A�*

epsilon  P�E��.       ��W�	N���ǩ�A�* 

Average reward per step  P�/oR       ��2	���ǩ�A�*

epsilon  P�����.       ��W�	����ǩ�A�* 

Average reward per step  P��n!       ��2	����ǩ�A�*

epsilon  P����.       ��W�	.���ǩ�A�* 

Average reward per step  P�.l�H       ��2	����ǩ�A�*

epsilon  P�vT8.       ��W�	���ǩ�A�* 

Average reward per step  P��D��       ��2	���ǩ�A�*

epsilon  P�6��-.       ��W�	y#��ǩ�A�* 

Average reward per step  P�����       ��2	W$��ǩ�A�*

epsilon  P�iE�.       ��W�	���ǩ�A�* 

Average reward per step  P��,�       ��2	ѕ��ǩ�A�*

epsilon  P��P!�.       ��W�	Ե��ǩ�A�* 

Average reward per step  P�*ڢ�       ��2	_���ǩ�A�*

epsilon  P�$��.       ��W�	���ǩ�A�* 

Average reward per step  P��L&       ��2	����ǩ�A�*

epsilon  P�f�g.       ��W�	]���ǩ�A�* 

Average reward per step  P�6%��       ��2	L���ǩ�A�*

epsilon  P����.       ��W�	���ǩ�A�* 

Average reward per step  P��GK0       ��2	��ǩ�A�*

epsilon  P��C��.       ��W�	�'��ǩ�A�* 

Average reward per step  P�Â�}       ��2	(��ǩ�A�*

epsilon  P�$a�.       ��W�	|��ǩ�A�* 

Average reward per step  P��Z4       ��2	���ǩ�A�*

epsilon  P��	U5.       ��W�	����ǩ�A�* 

Average reward per step  P�}��,       ��2	����ǩ�A�*

epsilon  P���.       ��W�	���ǩ�A�* 

Average reward per step  P��}��       ��2	��ǩ�A�*

epsilon  P�����.       ��W�	�"��ǩ�A�* 

Average reward per step  P��N�E       ��2	h#��ǩ�A�*

epsilon  P��.       ��W�	S"��ǩ�A�* 

Average reward per step  P�b��       ��2	�"��ǩ�A�*

epsilon  P�j��.       ��W�	�5��ǩ�A�* 

Average reward per step  P���c       ��2	a6��ǩ�A�*

epsilon  P�o�'�.       ��W�	�h��ǩ�A�* 

Average reward per step  P�l��       ��2	8i��ǩ�A�*

epsilon  P�-c��.       ��W�	\���ǩ�A�* 

Average reward per step  P����^       ��2	.���ǩ�A�*

epsilon  P��'�.       ��W�	���ǩ�A�* 

Average reward per step  P��Mv       ��2	����ǩ�A�*

epsilon  P���{�.       ��W�	 �ǩ�A�* 

Average reward per step  P���       ��2	� �ǩ�A�*

epsilon  P���4�.       ��W�	@3�ǩ�A�* 

Average reward per step  P����>       ��2	Y4�ǩ�A�*

epsilon  P�ۯ��.       ��W�	k��ǩ�A�* 

Average reward per step  P�=�r�       ��2	=��ǩ�A�*

epsilon  P����_.       ��W�	[��ǩ�A�* 

Average reward per step  P��þ       ��2	0��ǩ�A�*

epsilon  P��x�d.       ��W�	��ǩ�A�* 

Average reward per step  P�Zy�       ��2	��ǩ�A�*

epsilon  P�"��.       ��W�	�
�ǩ�A�* 

Average reward per step  P�(9(       ��2	
�ǩ�A�*

epsilon  P����8.       ��W�	�J�ǩ�A�* 

Average reward per step  P����       ��2	<K�ǩ�A�*

epsilon  P�bn�.       ��W�	��ǩ�A�* 

Average reward per step  P��       ��2	^��ǩ�A�*

epsilon  P�#�8h.       ��W�	R��ǩ�A�* 

Average reward per step  P����,       ��2	��ǩ�A�*

epsilon  P����t.       ��W�	��ǩ�A�* 

Average reward per step  P���'q       ��2	:�ǩ�A�*

epsilon  P��\ao.       ��W�	z��ǩ�A�* 

Average reward per step  P�:0�i       ��2	��ǩ�A�*

epsilon  P����t.       ��W�	m9�ǩ�A�* 

Average reward per step  P�c�t�       ��2	.:�ǩ�A�*

epsilon  P��c��.       ��W�	_�ǩ�A�* 

Average reward per step  P�� �       ��2	�_�ǩ�A�*

epsilon  P��Se.       ��W�	��ǩ�A�* 

Average reward per step  P��eV�       ��2	���ǩ�A�*

epsilon  P�+��.       ��W�	e��ǩ�A�* 

Average reward per step  P�e�lZ       ��2	���ǩ�A�*

epsilon  P�[�N.       ��W�	^�^�ǩ�A�* 

Average reward per step  P�矴�       ��2	��^�ǩ�A�*

epsilon  P�T΍�.       ��W�	��`�ǩ�A�* 

Average reward per step  P�F�       ��2	7�`�ǩ�A�*

epsilon  P�D��Z.       ��W�	��b�ǩ�A�* 

Average reward per step  P���͐       ��2	N�b�ǩ�A�*

epsilon  P����#.       ��W�	�d�ǩ�A�* 

Average reward per step  P����       ��2	��d�ǩ�A�*

epsilon  P��z�.       ��W�	�{f�ǩ�A�* 

Average reward per step  P��w�       ��2	$|f�ǩ�A�*

epsilon  P�Et�0.       ��W�	u�h�ǩ�A�* 

Average reward per step  P��S.       ��2	�h�ǩ�A�*

epsilon  P�%P�E.       ��W�	�j�ǩ�A�* 

Average reward per step  P�6�?       ��2	��j�ǩ�A�*

epsilon  P��Ɵ.       ��W�	��l�ǩ�A�* 

Average reward per step  P��\��       ��2	F�l�ǩ�A�*

epsilon  P�ah.       ��W�	)o�ǩ�A�* 

Average reward per step  P��~�       ��2	�)o�ǩ�A�*

epsilon  P���#.       ��W�	MLq�ǩ�A�* 

Average reward per step  P���d       ��2	Mq�ǩ�A�*

epsilon  P�<�[.       ��W�	Nzs�ǩ�A�* 

Average reward per step  P����        ��2	�zs�ǩ�A�*

epsilon  P�*�.       ��W�	*�u�ǩ�A�* 

Average reward per step  P��aqr       ��2	�u�ǩ�A�*

epsilon  P��*ְ.       ��W�	�Ew�ǩ�A�* 

Average reward per step  P�\�Ӹ       ��2	AFw�ǩ�A�*

epsilon  P�?���.       ��W�	3ky�ǩ�A�* 

Average reward per step  P��Hl       ��2		ly�ǩ�A�*

epsilon  P�>�\.       ��W�	�5|�ǩ�A�* 

Average reward per step  P�׾#�       ��2	�6|�ǩ�A�*

epsilon  P����.       ��W�	I�}�ǩ�A�* 

Average reward per step  P�$�       ��2	�}�ǩ�A�*

epsilon  P���,.       ��W�	���ǩ�A�* 

Average reward per step  P�q<,�       ��2	1��ǩ�A�*

epsilon  P����[.       ��W�	
��ǩ�A�* 

Average reward per step  P���4�       ��2	���ǩ�A�*

epsilon  P�Q��.       ��W�	Q��ǩ�A�* 

Average reward per step  P�'V�Q       ��2	��ǩ�A�*

epsilon  P�v5�.       ��W�	&6��ǩ�A�* 

Average reward per step  P�h�       ��2	�6��ǩ�A�*

epsilon  P�ߓT�.       ��W�	~���ǩ�A�* 

Average reward per step  P���z�       ��2	P���ǩ�A�*

epsilon  P����S.       ��W�	���ǩ�A�* 

Average reward per step  P���f�       ��2	���ǩ�A�*

epsilon  P�T.c.       ��W�	�'��ǩ�A�* 

Average reward per step  P��_�       ��2	,(��ǩ�A�*

epsilon  P�>"�0.       ��W�	cb��ǩ�A�* 

Average reward per step  P����       ��2	�b��ǩ�A�*

epsilon  P�����.       ��W�	z��ǩ�A�* 

Average reward per step  P���6�       ��2	q��ǩ�A�*

epsilon  P�YD1.       ��W�	>&��ǩ�A�* 

Average reward per step  P�K8a�       ��2	 '��ǩ�A�*

epsilon  P�+#5.       ��W�	0I��ǩ�A�* 

Average reward per step  P��^�       ��2	�I��ǩ�A�*

epsilon  P�3噑.       ��W�	���ǩ�A�* 

Average reward per step  P�l       ��2	���ǩ�A�*

epsilon  P��t�.       ��W�	��ǩ�A�* 

Average reward per step  P��Q�       ��2	���ǩ�A�*

epsilon  P��E�.       ��W�	�A��ǩ�A�* 

Average reward per step  P��$��       ��2	�B��ǩ�A�*

epsilon  P��j@.       ��W�	Ɯ�ǩ�A�* 

Average reward per step  P����       ��2	�Ɯ�ǩ�A�*

epsilon  P�����.       ��W�	�H��ǩ�A�* 

Average reward per step  P�O�I�       ��2	RI��ǩ�A�*

epsilon  P�'��0.       ��W�	Hp��ǩ�A�* 

Average reward per step  P�J5C�       ��2	q��ǩ�A�*

epsilon  P��U`.       ��W�	����ǩ�A�* 

Average reward per step  P�^=5       ��2	N���ǩ�A�*

epsilon  P���o�0       ���_	JϢ�ǩ�AY*#
!
Average reward per episode,������.       ��W�	�Ϣ�ǩ�AY*!

total reward per episode   �1�.       ��W�	���ǩ�A�* 

Average reward per step,�����=       ��2	j��ǩ�A�*

epsilon,���%L.       ��W�	�D��ǩ�A�* 

Average reward per step,��1,�&       ��2	_E��ǩ�A�*

epsilon,��,_��.       ��W�	��ǩ�A�* 

Average reward per step,��}��g       ��2	���ǩ�A�*

epsilon,��4���.       ��W�	Uj��ǩ�A�* 

Average reward per step,��୓B       ��2	#k��ǩ�A�*

epsilon,��6�.�.       ��W�	/���ǩ�A�* 

Average reward per step,��Q�>       ��2	���ǩ�A�*

epsilon,����.       ��W�	�ܱ�ǩ�A�* 

Average reward per step,����bz       ��2	�ݱ�ǩ�A�*

epsilon,���V�.       ��W�	�P��ǩ�A�* 

Average reward per step,����v�       ��2	�Q��ǩ�A�*

epsilon,���b�f.       ��W�	���ǩ�A�* 

Average reward per step,���-�       ��2	܄��ǩ�A�*

epsilon,��q�e.       ��W�	ڬ��ǩ�A�* 

Average reward per step,���A&       ��2	m���ǩ�A�*

epsilon,���ˋ.       ��W�	¹�ǩ�A�* 

Average reward per step,��ԣ%       ��2	�¹�ǩ�A�*

epsilon,����fa.       ��W�	}��ǩ�A�* 

Average reward per step,����~;       ��2	`��ǩ�A�*

epsilon,��5p�.       ��W�	�~��ǩ�A�* 

Average reward per step,��� 3�       ��2	���ǩ�A�*

epsilon,�����W.       ��W�	!���ǩ�A�* 

Average reward per step,���ه(       ��2	����ǩ�A�*

epsilon,��HFo�.       ��W�	����ǩ�A�* 

Average reward per step,����ؖ       ��2	����ǩ�A�*

epsilon,�� ��	.       ��W�	���ǩ�A�* 

Average reward per step,������       ��2	���ǩ�A�*

epsilon,��/�b�.       ��W�	3O��ǩ�A�* 

Average reward per step,����W�       ��2	�O��ǩ�A�*

epsilon,��^޽�.       ��W�	����ǩ�A�* 

Average reward per step,���]�3       ��2	g���ǩ�A�*

epsilon,��r���.       ��W�	���ǩ�A�* 

Average reward per step,�����       ��2	v��ǩ�A�*

epsilon,��:E�.       ��W�	4-��ǩ�A�* 

Average reward per step,��G.�       ��2	A.��ǩ�A�*

epsilon,��~���0       ���_	�J��ǩ�AZ*#
!
Average reward per episode(���c?j.       ��W�	+K��ǩ�AZ*!

total reward per episode  ����.       ��W�	OX��ǩ�A�* 

Average reward per step(��Z��       ��2	%Y��ǩ�A�*

epsilon(��ʥ	k.       ��W�	����ǩ�A�* 

Average reward per step(��`L       ��2	����ǩ�A�*

epsilon(���XF/.       ��W�	����ǩ�A�* 

Average reward per step(��dK,�       ��2	@���ǩ�A�*

epsilon(��-kM.       ��W�	�=��ǩ�A�* 

Average reward per step(��_�4       ��2	`>��ǩ�A�*

epsilon(��Fn^.       ��W�	^��ǩ�A�* 

Average reward per step(��O��       ��2	�^��ǩ�A�*

epsilon(��.�0�.       ��W�	k���ǩ�A�* 

Average reward per step(���׊       ��2	0���ǩ�A�*

epsilon(����.       ��W�	���ǩ�A�* 

Average reward per step(��e�f7       ��2	���ǩ�A�*

epsilon(��;^KT.       ��W�	c���ǩ�A�* 

Average reward per step(��E�~�       ��2	���ǩ�A�*

epsilon(��਄.       ��W�	6w��ǩ�A�* 

Average reward per step(���k�       ��2	�w��ǩ�A�*

epsilon(��w}.       ��W�	����ǩ�A�* 

Average reward per step(����       ��2	8���ǩ�A�*

epsilon(��-��B.       ��W�	� ��ǩ�A�* 

Average reward per step(��a*�       ��2	���ǩ�A�*

epsilon(���1.       ��W�	S@��ǩ�A�* 

Average reward per step(���dN�       ��2	�@��ǩ�A�*

epsilon(��|�~.       ��W�	����ǩ�A�* 

Average reward per step(���8-)       ��2	[���ǩ�A�*

epsilon(���T.       ��W�	����ǩ�A�* 

Average reward per step(���6g       ��2	����ǩ�A�*

epsilon(���a� .       ��W�	��ǩ�A�* 

Average reward per step(���j�M       ��2	���ǩ�A�*

epsilon(���̣�.       ��W�	�G��ǩ�A�* 

Average reward per step(��ë       ��2	�H��ǩ�A�*

epsilon(���G�v.       ��W�	l��ǩ�A�* 

Average reward per step(��1�-       ��2	�l��ǩ�A�*

epsilon(���zz�.       ��W�	ML��ǩ�A�* 

Average reward per step(���3��       ��2	M��ǩ�A�*

epsilon(��T�'9.       ��W�	����ǩ�A�* 

Average reward per step(��Bڧ       ��2	����ǩ�A�*

epsilon(���R�0       ���_	i��ǩ�A[*#
!
Average reward per episode���{1!.       ��W�	���ǩ�A[*!

total reward per episode  &�)��.       ��W�	����ǩ�A�* 

Average reward per step���Ѕ&       ��2	w���ǩ�A�*

epsilon�����(.       ��W�	�h��ǩ�A�* 

Average reward per step��/Ь       ��2	�i��ǩ�A�*

epsilon���o�.       ��W�	L���ǩ�A�* 

Average reward per step���7��       ��2	���ǩ�A�*

epsilon���p�.       ��W�	���ǩ�A�* 

Average reward per step��iϊ�       ��2	���ǩ�A�*

epsilon��0��J.       ��W�	ձ�ǩ�A�* 

Average reward per step����y�       ��2	l��ǩ�A�*

epsilon��bwWt.       ��W�	Z��ǩ�A�* 

Average reward per step��h�*       ��2	8��ǩ�A�*

epsilon��y#�.       ��W�	G��ǩ�A�* 

Average reward per step���0�l       ��2	���ǩ�A�*

epsilon�����).       ��W�	H�	�ǩ�A�* 

Average reward per step��$S�       ��2	&�	�ǩ�A�*

epsilon���AU.       ��W�	��ǩ�A�* 

Average reward per step��X��       ��2	��ǩ�A�*

epsilon���w7�.       ��W�	s+�ǩ�A�* 

Average reward per step��i��       ��2	4,�ǩ�A�*

epsilon��U�d�.       ��W�	c��ǩ�A�* 

Average reward per step����%�       ��2	F��ǩ�A�*

epsilon���BBF.       ��W�	,�ǩ�A�* 

Average reward per step���v�       ��2	�,�ǩ�A�*

epsilon���u_�.       ��W�	�W�ǩ�A�* 

Average reward per step��m��@       ��2	�X�ǩ�A�*

epsilon���C��.       ��W�	^��ǩ�A�* 

Average reward per step����n       ��2	4��ǩ�A�*

epsilon��a�Ւ.       ��W�	��ǩ�A�* 

Average reward per step���'=>       ��2	��ǩ�A�*

epsilon��%���.       ��W�	(~�ǩ�A�* 

Average reward per step��GF�U       ��2	�ǩ�A�*

epsilon��.Yګ.       ��W�	B�ǩ�A�* 

Average reward per step�����i       ��2	�B�ǩ�A�*

epsilon��R>�0       ���_	fi�ǩ�A\*#
!
Average reward per episode�'8).       ��W�	+j�ǩ�A\*!

total reward per episode  ��"�<.       ��W�	��_�ǩ�A�* 

Average reward per step��
�
       ��2	��_�ǩ�A�*

epsilon����A.       ��W�	0�a�ǩ�A�* 

Average reward per step�L��       ��2	#�a�ǩ�A�*

epsilon�5�>c.       ��W�	��c�ǩ�A�* 

Average reward per step� �|.       ��2	C�c�ǩ�A�*

epsilon��6o�.       ��W�	��e�ǩ�A�* 

Average reward per step����       ��2	� f�ǩ�A�*

epsilon�I�.       ��W�	��g�ǩ�A�* 

Average reward per step�U�c       ��2	p�g�ǩ�A�*

epsilon�)6V�.       ��W�	h�i�ǩ�A�* 

Average reward per step�w�=�       ��2	S�i�ǩ�A�*

epsilon����.       ��W�	�
l�ǩ�A�* 

Average reward per step����       ��2	�l�ǩ�A�*

epsilon���9>.       ��W�	>@n�ǩ�A�* 

Average reward per step��{l       ��2	�@n�ǩ�A�*

epsilon���)�.       ��W�	��o�ǩ�A�* 

Average reward per step�M�t       ��2	��o�ǩ�A�*

epsilon�cIv].       ��W�	��q�ǩ�A�* 

Average reward per step����g       ��2	��q�ǩ�A�*

epsilon�FG?.       ��W�	U0t�ǩ�A�* 

Average reward per step����d       ��2	�0t�ǩ�A�*

epsilon�KP�w.       ��W�	�wv�ǩ�A�* 

Average reward per step�<cO^       ��2	�zv�ǩ�A�*

epsilon��h�.       ��W�	Hx�ǩ�A�* 

Average reward per step�%�Ғ       ��2	�x�ǩ�A�*

epsilon��#�c.       ��W�	eTz�ǩ�A�* 

Average reward per step��       ��2	2Uz�ǩ�A�*

epsilon���N.       ��W�	��|�ǩ�A�* 

Average reward per step�ԑ��       ��2	Q�|�ǩ�A�*

epsilon�FչM.       ��W�	`~�ǩ�A�* 

Average reward per step�֭o       ��2	S~�ǩ�A�*

epsilon�VSy.       ��W�	�&��ǩ�A�* 

Average reward per step�y���       ��2	1(��ǩ�A�*

epsilon��.       ��W�	�P��ǩ�A�* 

Average reward per step�&S;Z       ��2	�Q��ǩ�A�*

epsilon�V}H.       ��W�	��ǩ�A�* 

Average reward per step��V�       ��2	����ǩ�A�*

epsilon�H9b.       ��W�	����ǩ�A�* 

Average reward per step��$�/       ��2	Ի��ǩ�A�*

epsilon��4y.       ��W�	u9��ǩ�A�* 

Average reward per step�q�W�       ��2	C:��ǩ�A�*

epsilon� �}�.       ��W�	e��ǩ�A�* 

Average reward per step�ȶ[       ��2	jg��ǩ�A�*

epsilon����.       ��W�	/���ǩ�A�* 

Average reward per step��8s       ��2	7���ǩ�A�*

epsilon��48b.       ��W�	�Վ�ǩ�A�* 

Average reward per step��1��       ��2	k֎�ǩ�A�*

epsilon��o�.       ��W�	�V��ǩ�A�* 

Average reward per step� ���       ��2	�W��ǩ�A�*

epsilon���F�.       ��W�	`u��ǩ�A�* 

Average reward per step�8�       ��2	v��ǩ�A�*

epsilon���a�.       ��W�	����ǩ�A�* 

Average reward per step�Yu��       ��2	P���ǩ�A�*

epsilon�d���.       ��W�	~��ǩ�A�* 

Average reward per step��Ȓ�       ��2	h��ǩ�A�*

epsilon�@,��.       ��W�	��ǩ�A�* 

Average reward per step�p�d�       ��2	���ǩ�A�*

epsilon�Rt%.       ��W�	����ǩ�A�* 

Average reward per step����       ��2	����ǩ�A�*

epsilon��#<�0       ���_	m���ǩ�A]*#
!
Average reward per episode�����ɫ.       ��W�	����ǩ�A]*!

total reward per episode  �*)h�.       ��W�	����ǩ�A�* 

Average reward per step����O��0       ��2	���ǩ�A�*

epsilon�����e{.       ��W�	_'��ǩ�A�* 

Average reward per step�����/��       ��2	,(��ǩ�A�*

epsilon�����e<.       ��W�	I��ǩ�A�* 

Average reward per step����?S,       ��2	�I��ǩ�A�*

epsilon�����l.       ��W�	���ǩ�A�* 

Average reward per step������D�       ��2	����ǩ�A�*

epsilon������Ղ.       ��W�	{��ǩ�A�* 

Average reward per step����r�U�       ��2	M��ǩ�A�*

epsilon����N�q�.       ��W�	�]��ǩ�A�* 

Average reward per step����:2'�       ��2	�^��ǩ�A�*

epsilon�����y<�.       ��W�	��ǩ�A�* 

Average reward per step����k
=�       ��2	���ǩ�A�*

epsilon����[h�.       ��W�	殭�ǩ�A�* 

Average reward per step�����       ��2	ޯ��ǩ�A�*

epsilon�����r�0.       ��W�	;��ǩ�A�* 

Average reward per step����{YJu       ��2	�;��ǩ�A�*

epsilon����+�V9.       ��W�	�s��ǩ�A�* 

Average reward per step�����ˢ       ��2	�t��ǩ�A�*

epsilon�����Tc.       ��W�	S���ǩ�A�* 

Average reward per step����H�v       ��2	B���ǩ�A�*

epsilon����n���.       ��W�	Aص�ǩ�A�* 

Average reward per step�����$��       ��2	�ص�ǩ�A�*

epsilon������j9.       ��W�	[`��ǩ�A�* 

Average reward per step�����K�       ��2	Ja��ǩ�A�*

epsilon�����E�.       ��W�	臹�ǩ�A�* 

Average reward per step�����,y       ��2	ƈ��ǩ�A�*

epsilon������v.       ��W�	����ǩ�A�* 

Average reward per step����&�ٌ       ��2	����ǩ�A�*

epsilon��������.       ��W�	�߽�ǩ�A�* 

Average reward per step����G��       ��2	���ǩ�A�*

epsilon������j�.       ��W�	����ǩ�A�* 

Average reward per step������3�       ��2	����ǩ�A�*

epsilon����3�>A.       ��W�	^���ǩ�A�* 

Average reward per step������58       ��2	���ǩ�A�*

epsilon�����xi�.       ��W�	����ǩ�A�* 

Average reward per step����H3)2       ��2	����ǩ�A�*

epsilon����Q�.       ��W�	L���ǩ�A�* 

Average reward per step�����T       ��2	���ǩ�A�*

epsilon�����.       ��W�	:��ǩ�A�* 

Average reward per step������o�       ��2	)	��ǩ�A�*

epsilon�������.       ��W�	m���ǩ�A�* 

Average reward per step����U3��       ��2	\���ǩ�A�*

epsilon������cO.       ��W�	�Q��ǩ�A�* 

Average reward per step�����B�'       ��2	�R��ǩ�A�*

epsilon������4�.       ��W�	u���ǩ�A�* 

Average reward per step����y��       ��2	m���ǩ�A�*

epsilon�����;�.       ��W�	�2��ǩ�A�* 

Average reward per step�����:��       ��2	�3��ǩ�A�*

epsilon����^���.       ��W�	NF��ǩ�A�* 

Average reward per step�����       ��2	�F��ǩ�A�*

epsilon�����+��.       ��W�	S���ǩ�A�* 

Average reward per step����9Ib�       ��2	���ǩ�A�*

epsilon����]d�.       ��W�	]��ǩ�A�* 

Average reward per step�����⓯       ��2	@��ǩ�A�*

epsilon�������:.       ��W�	�V�ǩ�A�* 

Average reward per step������v�       ��2	�W�ǩ�A�*

epsilon������.       ��W�	{��ǩ�A�* 

Average reward per step�������       ��2	s��ǩ�A�*

epsilon������"�.       ��W�	֩�ǩ�A�* 

Average reward per step����3Χ�       ��2	z��ǩ�A�*

epsilon�����W.       ��W�	�� �ǩ�A�* 

Average reward per step�������@       ��2	n� �ǩ�A�*

epsilon����H<|n.       ��W�	:]"�ǩ�A�* 

Average reward per step������       ��2	^"�ǩ�A�*

epsilon����@��.       ��W�	
�$�ǩ�A�* 

Average reward per step�����34�       ��2	��$�ǩ�A�*

epsilon������g�.       ��W�	A�&�ǩ�A�* 

Average reward per step����oǹ�       ��2	��&�ǩ�A�*

epsilon�����
n�.       ��W�	�)�ǩ�A�* 

Average reward per step����vA��       ��2	�)�ǩ�A�*

epsilon�����w�d.       ��W�	P�*�ǩ�A�* 

Average reward per step����F�2q       ��2	.�*�ǩ�A�*

epsilon����pg3.       ��W�	մ,�ǩ�A�* 

Average reward per step�����4�       ��2	|�,�ǩ�A�*

epsilon����'�.       ��W�	?�.�ǩ�A�* 

Average reward per step���� P	6       ��2	&�.�ǩ�A�*

epsilon����ŗ_Q.       ��W�	�1�ǩ�A�* 

Average reward per step������v       ��2	�1�ǩ�A�*

epsilon����Ȭc`.       ��W�	�R3�ǩ�A�* 

Average reward per step�����=       ��2	]S3�ǩ�A�*

epsilon����g4�.       ��W�	��4�ǩ�A�* 

Average reward per step����3�&       ��2	��4�ǩ�A�*

epsilon�����R�+.       ��W�	NB7�ǩ�A�* 

Average reward per step����&��8       ��2	9C7�ǩ�A�*

epsilon����Nyޠ.       ��W�	Oy9�ǩ�A�* 

Average reward per step����#oc       ��2	1z9�ǩ�A�*

epsilon����1�]�.       ��W�	_�:�ǩ�A�* 

Average reward per step����Nj-�       ��2	W�:�ǩ�A�*

epsilon��������.       ��W�	=�ǩ�A�* 

Average reward per step����Fz�d       ��2	�=�ǩ�A�*

epsilon���� �<30       ���_	�;=�ǩ�A^*#
!
Average reward per episode�M�m�6'.       ��W�	h<=�ǩ�A^*!

total reward per episode  �����*.       ��W�	A�A�ǩ�A�* 

Average reward per step�M��Zgl       ��2	�A�ǩ�A�*

epsilon�M�<�ko.       ��W�	��C�ǩ�A�* 

Average reward per step�M�Ʉ<�       ��2	��C�ǩ�A�*

epsilon�M��%��.       ��W�	h^E�ǩ�A�* 

Average reward per step�M���Б       ��2	=_E�ǩ�A�*

epsilon�M�+���.       ��W�	5|G�ǩ�A�* 

Average reward per step�M�:��M       ��2	}G�ǩ�A�*

epsilon�M�"�@�.       ��W�	�EJ�ǩ�A�* 

Average reward per step�M�x�pa       ��2	gFJ�ǩ�A�*

epsilon�M����U.       ��W�	��K�ǩ�A�* 

Average reward per step�M�6P��       ��2	[�K�ǩ�A�*

epsilon�M�h.       ��W�	�M�ǩ�A�* 

Average reward per step�M�ǡ�       ��2	��M�ǩ�A�*

epsilon�M�H���.       ��W�	��O�ǩ�A�* 

Average reward per step�M�/r}       ��2	��O�ǩ�A�*

epsilon�M�����.       ��W�	6R�ǩ�A�* 

Average reward per step�M���	       ��2	�6R�ǩ�A�*

epsilon�M���^j.       ��W�	��S�ǩ�A�* 

Average reward per step�M�덁�       ��2	_�S�ǩ�A�*

epsilon�M��8}�.       ��W�	��U�ǩ�A�* 

Average reward per step�M�I�       ��2	��U�ǩ�A�*

epsilon�M��4&V.       ��W�	��W�ǩ�A�* 

Average reward per step�M���:�       ��2	��W�ǩ�A�*

epsilon�M���~.       ��W�	ۿY�ǩ�A�* 

Average reward per step�M�uZ��       ��2	��Y�ǩ�A�*

epsilon�M���ٕ.       ��W�	@�[�ǩ�A�* 

Average reward per step�M��w"�       ��2	�[�ǩ�A�*

epsilon�M�u��H.       ��W�	PR^�ǩ�A�* 

Average reward per step�M���oL       ��2	�R^�ǩ�A�*

epsilon�M���x�.       ��W�	Hm`�ǩ�A�* 

Average reward per step�M�4��       ��2	Ln`�ǩ�A�*

epsilon�M�S=]�.       ��W�	�b�ǩ�A�* 

Average reward per step�M����       ��2	�b�ǩ�A�*

epsilon�M��R7<.       ��W�	o�f�ǩ�A�* 

Average reward per step�M���2o       ��2	�f�ǩ�A�*

epsilon�M�ߑ�8.       ��W�	��h�ǩ�A�* 

Average reward per step�M��ة       ��2	)�h�ǩ�A�*

epsilon�M�����.       ��W�	^j�ǩ�A�* 

Average reward per step�M��zA       ��2	�^j�ǩ�A�*

epsilon�M�� �.       ��W�	Кl�ǩ�A�* 

Average reward per step�M�k�       ��2	��l�ǩ�A�*

epsilon�M�j�bw.       ��W�	��n�ǩ�A�* 

Average reward per step�M��Dr       ��2	�n�ǩ�A�*

epsilon�M�����.       ��W�	;Vp�ǩ�A�* 

Average reward per step�M��|�       ��2	�Vp�ǩ�A�*

epsilon�M��p��.       ��W�	r�ǩ�A�* 

Average reward per step�M�/r�       ��2	��r�ǩ�A�*

epsilon�M�*}l�0       ���_	��r�ǩ�A_*#
!
Average reward per episode  �����.       ��W�	�r�ǩ�A_*!

total reward per episode  �
=.       ��W�	}�v�ǩ�A�* 

Average reward per step  ��>�3       ��2	g�v�ǩ�A�*

epsilon  ��+lI.       ��W�	��x�ǩ�A�* 

Average reward per step  ��L�4�       ��2	7�x�ǩ�A�*

epsilon  ��*&.       ��W�	�z�ǩ�A�* 

Average reward per step  ��?BМ       ��2	��z�ǩ�A�*

epsilon  �����.       ��W�	j}�ǩ�A�* 

Average reward per step  ���?�-       ��2	}�ǩ�A�*

epsilon  ���/��.       ��W�	;�~�ǩ�A�* 

Average reward per step  ��B%;�       ��2	�~�ǩ�A�*

epsilon  �����,.       ��W�	��ǩ�A�* 

Average reward per step  ���Vz       ��2	���ǩ�A�*

epsilon  ��?m�.       ��W�	�	��ǩ�A�* 

Average reward per step  ��le�'       ��2	�
��ǩ�A�*

epsilon  ����.       ��W�	�F��ǩ�A�* 

Average reward per step  ��weN       ��2	�G��ǩ�A�*

epsilon  ���@B.       ��W�	�S��ǩ�A�* 

Average reward per step  ����4#       ��2	*T��ǩ�A�*

epsilon  ��� �J.       ��W�	�|��ǩ�A�* 

Average reward per step  ���f�H       ��2	�}��ǩ�A�*

epsilon  ��Y|��.       ��W�	�2��ǩ�A�* 

Average reward per step  ��^#��       ��2	Q3��ǩ�A�*

epsilon  ��*��1.       ��W�	�b��ǩ�A�* 

Average reward per step  ����{P       ��2	�c��ǩ�A�*

epsilon  ��2)�.       ��W�	���ǩ�A�* 

Average reward per step  ������       ��2	����ǩ�A�*

epsilon  ��^�+�.       ��W�	亓�ǩ�A�* 

Average reward per step  �����       ��2	����ǩ�A�*

epsilon  ��K�O.       ��W�	�5��ǩ�A�* 

Average reward per step  ������       ��2	�6��ǩ�A�*

epsilon  ����J�.       ��W�	�o��ǩ�A�* 

Average reward per step  ��@KД       ��2	�p��ǩ�A�*

epsilon  ��M'�.       ��W�	���ǩ�A�* 

Average reward per step  ���k       ��2	����ǩ�A�*

epsilon  ����N.       ��W�	�Ǜ�ǩ�A�* 

Average reward per step  ���n�       ��2	2ț�ǩ�A�*

epsilon  �����.       ��W�	S[��ǩ�A�* 

Average reward per step  ��6��0       ��2	\��ǩ�A�*

epsilon  ���7�I.       ��W�	ߋ��ǩ�A�* 

Average reward per step  ��,�SN       ��2	~���ǩ�A�*

epsilon  ��áT�.       ��W�	����ǩ�A�* 

Average reward per step  ���W��       ��2	k���ǩ�A�*

epsilon  ���rx.       ��W�	���ǩ�A�* 

Average reward per step  ���r��       ��2	T��ǩ�A�*

epsilon  ��F�n}.       ��W�	����ǩ�A�* 

Average reward per step  ��J��       ��2	����ǩ�A�*

epsilon  ���QR�.       ��W�	U���ǩ�A�* 

Average reward per step  ���)�       ��2	3���ǩ�A�*

epsilon  �����J.       ��W�	;��ǩ�A�* 

Average reward per step  ����n       ��2	���ǩ�A�*

epsilon  ��a��4.       ��W�	΋��ǩ�A�* 

Average reward per step  ��r�       ��2	����ǩ�A�*

epsilon  ��ɶ��.       ��W�	 ���ǩ�A�* 

Average reward per step  ���v�       ��2	 ���ǩ�A�*

epsilon  ����S.       ��W�	�
��ǩ�A�* 

Average reward per step  ��B[�{       ��2	���ǩ�A�*

epsilon  ��Qz��.       ��W�	C;��ǩ�A�* 

Average reward per step  ����*       ��2	�;��ǩ�A�*

epsilon  �����).       ��W�	o��ǩ�A�* 

Average reward per step  ��}jÛ       ��2	I��ǩ�A�*

epsilon  ���7D�.       ��W�	���ǩ�A�* 

Average reward per step  ����L�       ��2	� ��ǩ�A�*

epsilon  ����`m.       ��W�	�[��ǩ�A�* 

Average reward per step  ��'�       ��2	�\��ǩ�A�*

epsilon  ��Y13�.       ��W�	Z��ǩ�A�* 

Average reward per step  ���ɗ       ��2	=��ǩ�A�*

epsilon  ��:���.       ��W�	
/��ǩ�A�* 

Average reward per step  ��5<�       ��2	�/��ǩ�A�*

epsilon  ��L���.       ��W�	�9��ǩ�A�* 

Average reward per step  ���]^       ��2	K:��ǩ�A�*

epsilon  ���Er.       ��W�	�N��ǩ�A�* 

Average reward per step  ���X0       ��2	YO��ǩ�A�*

epsilon  ����s�.       ��W�	 q��ǩ�A�* 

Average reward per step  ���>&�       ��2	�q��ǩ�A�*

epsilon  ����&*.       ��W�	����ǩ�A�* 

Average reward per step  ���r~*       ��2	{���ǩ�A�*

epsilon  ��'�l.       ��W�	&S��ǩ�A�* 

Average reward per step  ���bQ�       ��2	T��ǩ�A�*

epsilon  ��iq�.       ��W�	b���ǩ�A�* 

Average reward per step  ���7R       ��2	E���ǩ�A�*

epsilon  ��<j6�.       ��W�	ۧ��ǩ�A�* 

Average reward per step  ��_vg�       ��2	����ǩ�A�*

epsilon  �����=.       ��W�	N���ǩ�A�* 

Average reward per step  ���E�       ��2	���ǩ�A�*

epsilon  ��.��*.       ��W�	S���ǩ�A�* 

Average reward per step  �����/       ��2	����ǩ�A�*

epsilon  ���4.       ��W�	k��ǩ�A�* 

Average reward per step  ���+q�       ��2	<��ǩ�A�*

epsilon  ���;|n.       ��W�	f���ǩ�A�* 

Average reward per step  ��M���       ��2	P���ǩ�A�*

epsilon  ��a%N.       ��W�	����ǩ�A�* 

Average reward per step  ��T�0       ��2	O���ǩ�A�*

epsilon  ��:�л.       ��W�	����ǩ�A�* 

Average reward per step  �����       ��2	{���ǩ�A�*

epsilon  ���u��.       ��W�	����ǩ�A�* 

Average reward per step  ���cb�       ��2	����ǩ�A�*

epsilon  ��Y�Ą.       ��W�	k���ǩ�A�* 

Average reward per step  �����h       ��2	���ǩ�A�*

epsilon  ��HV�.       ��W�	����ǩ�A�* 

Average reward per step  ���Ft�       ��2	'���ǩ�A�*

epsilon  ��,�.       ��W�	�C��ǩ�A�* 

Average reward per step  ���]�{       ��2	�D��ǩ�A�*

epsilon  ��,4qd.       ��W�	i��ǩ�A�* 

Average reward per step  ��q�E�       ��2	j��ǩ�A�*

epsilon  ����_;.       ��W�	S���ǩ�A�* 

Average reward per step  ���qX�       ��2	B���ǩ�A�*

epsilon  ��v��B.       ��W�	�I��ǩ�A�* 

Average reward per step  ��V�D       ��2	�J��ǩ�A�*

epsilon  �����i.       ��W�	���ǩ�A�* 

Average reward per step  ��_c��       ��2	A���ǩ�A�*

epsilon  ��_]�}.       ��W�	 ��ǩ�A�* 

Average reward per step  ���'M       ��2	���ǩ�A�*

epsilon  ��a�d5.       ��W�	�G��ǩ�A�* 

Average reward per step  ���KF       ��2	�H��ǩ�A�*

epsilon  ��ԑ8�.       ��W�	^c��ǩ�A�* 

Average reward per step  �����       ��2	Rd��ǩ�A�*

epsilon  ��"X��.       ��W�	���ǩ�A�* 

Average reward per step  ��K/��       ��2	����ǩ�A�*

epsilon  ��,_��.       ��W�	����ǩ�A�* 

Average reward per step  ���1��       ��2	[���ǩ�A�*

epsilon  ����"w.       ��W�	?���ǩ�A�* 

Average reward per step  ��`�{�       ��2	����ǩ�A�*

epsilon  ����N�.       ��W�	�w��ǩ�A�* 

Average reward per step  ��"_�b       ��2	�x��ǩ�A�*

epsilon  ��̓B.       ��W�	|G��ǩ�A�* 

Average reward per step  ���,��       ��2	(H��ǩ�A�*

epsilon  ��%d�.       ��W�	����ǩ�A�* 

Average reward per step  ��Ob,       ��2	����ǩ�A�*

epsilon  ����r.       ��W�	���ǩ�A�* 

Average reward per step  ��41�h       ��2	���ǩ�A�*

epsilon  ����� .       ��W�	x���ǩ�A�* 

Average reward per step  ��C���       ��2	A���ǩ�A�*

epsilon  ��ߕ�.       ��W�	}���ǩ�A�* 

Average reward per step  ��}ż�       ��2	)���ǩ�A�*

epsilon  ��ܡ<�.       ��W�	�ǩ�A�* 

Average reward per step  ����xM       ��2	��ǩ�A�*

epsilon  ����N�.       ��W�	�x�ǩ�A�* 

Average reward per step  ��*1��       ��2	�y�ǩ�A�*

epsilon  ���V��0       ���_	g��ǩ�A`*#
!
Average reward per episode����� .       ��W�	��ǩ�A`*!

total reward per episode  ��s.       ��W�	�Z
�ǩ�A�* 

Average reward per step���ͥ��       ��2	W[
�ǩ�A�*

epsilon���� L~.       ��W�	��ǩ�A�* 

Average reward per step����BB       ��2	�ǩ�A�*

epsilon������,.       ��W�	���ǩ�A�* 

Average reward per step���
-g       ��2	g��ǩ�A�*

epsilon���Z�.       ��W�	��ǩ�A�* 

Average reward per step������       ��2	���ǩ�A�*

epsilon���p(ӯ.       ��W�	. �ǩ�A�* 

Average reward per step����l�w       ��2	�"�ǩ�A�*

epsilon���8Ja.       ��W�	.<�ǩ�A�* 

Average reward per step���U��       ��2	=�ǩ�A�*

epsilon���tzFp.       ��W�	�^�ǩ�A�* 

Average reward per step���=M�m       ��2	9_�ǩ�A�*

epsilon���s2!q.       ��W�	���ǩ�A�* 

Average reward per step���QE�       ��2	���ǩ�A�*

epsilon���ѵ�.       ��W�	���ǩ�A�* 

Average reward per step�����E       ��2	Ǽ�ǩ�A�*

epsilon���X�0d.       ��W�	���ǩ�A�* 

Average reward per step���MX�       ��2	���ǩ�A�*

epsilon����<ڭ.       ��W�	�u�ǩ�A�* 

Average reward per step�����       ��2	�v�ǩ�A�*

epsilon���0��N.       ��W�	���ǩ�A�* 

Average reward per step����v+�       ��2	���ǩ�A�*

epsilon����wr).       ��W�	� "�ǩ�A�* 

Average reward per step����k�       ��2	�!"�ǩ�A�*

epsilon���=�aU.       ��W�	��#�ǩ�A�* 

Average reward per step���rď,       ��2	u�#�ǩ�A�*

epsilon�����+.       ��W�	�@&�ǩ�A�* 

Average reward per step����B�       ��2	�A&�ǩ�A�*

epsilon���@�6Y.       ��W�	��(�ǩ�A�* 

Average reward per step���R��       ��2	x�(�ǩ�A�*

epsilon�����v.       ��W�	�*�ǩ�A�* 

Average reward per step����s��       ��2	�*�ǩ�A�*

epsilon����D�.       ��W�	IK,�ǩ�A�* 

Average reward per step���4��       ��2	L,�ǩ�A�*

epsilon����gP�.       ��W�	 �.�ǩ�A�* 

Average reward per step�����]       ��2	ͭ.�ǩ�A�*

epsilon����G��.       ��W�	$0�ǩ�A�* 

Average reward per step���k�E�       ��2	%0�ǩ�A�*

epsilon����"�T.       ��W�	h2�ǩ�A�* 

Average reward per step�������       ��2	�h2�ǩ�A�*

epsilon���̜T.       ��W�	��4�ǩ�A�* 

Average reward per step����\       ��2	��4�ǩ�A�*

epsilon����Yy�.       ��W�	��6�ǩ�A�* 

Average reward per step����9{       ��2	��6�ǩ�A�*

epsilon���J���.       ��W�	rn8�ǩ�A�* 

Average reward per step����}Y       ��2	Po8�ǩ�A�*

epsilon���}��r.       ��W�	�:�ǩ�A�* 

Average reward per step���AOg       ��2	��:�ǩ�A�*

epsilon���2�Ѵ.       ��W�	��<�ǩ�A�* 

Average reward per step�����;       ��2	��<�ǩ�A�*

epsilon���a6X�.       ��W�	�?�ǩ�A�* 

Average reward per step���H���       ��2	�?�ǩ�A�*

epsilon���t�%.       ��W�	��@�ǩ�A�* 

Average reward per step�����O_       ��2	[�@�ǩ�A�*

epsilon����FN.       ��W�	�B�ǩ�A�* 

Average reward per step����#z�       ��2	��B�ǩ�A�*

epsilon����SX30       ���_	pC�ǩ�Aa*#
!
Average reward per episode�i��U�<�.       ��W�	�C�ǩ�Aa*!

total reward per episode  ���Є�.       ��W�	��F�ǩ�A�* 

Average reward per step�i��2n��       ��2	��F�ǩ�A�*

epsilon�i��"�/.       ��W�	��H�ǩ�A�* 

Average reward per step�i��2��       ��2	p�H�ǩ�A�*

epsilon�i��S
l.       ��W�		K�ǩ�A�* 

Average reward per step�i���fX       ��2	�	K�ǩ�A�*

epsilon�i��/�Š.       ��W�	('M�ǩ�A�* 

Average reward per step�i������       ��2	�'M�ǩ�A�*

epsilon�i��7q�.       ��W�	UO�ǩ�A�* 

Average reward per step�i��ʞ�T       ��2	�UO�ǩ�A�*

epsilon�i���}K.       ��W�	��Q�ǩ�A�* 

Average reward per step�i���m>�       ��2	V�Q�ǩ�A�*

epsilon�i���ʬ�.       ��W�	S�ǩ�A�* 

Average reward per step�i����4�       ��2	�S�ǩ�A�*

epsilon�i����
.       ��W�	�$U�ǩ�A�* 

Average reward per step�i���zYq       ��2	�%U�ǩ�A�*

epsilon�i����.       ��W�	3OW�ǩ�A�* 

Average reward per step�i��V��       ��2	PW�ǩ�A�*

epsilon�i��#��.       ��W�	J�Y�ǩ�A�* 

Average reward per step�i�����d       ��2	�Y�ǩ�A�*

epsilon�i��b(�q.       ��W�	�\[�ǩ�A�* 

Average reward per step�i��08�9       ��2	�][�ǩ�A�*

epsilon�i��ҭ��.       ��W�	Z�]�ǩ�A�* 

Average reward per step�i��R�\�       ��2	+�]�ǩ�A�*

epsilon�i��9J+@.       ��W�	��_�ǩ�A�* 

Average reward per step�i��vߍY       ��2	��_�ǩ�A�*

epsilon�i���!.       ��W�	�wa�ǩ�A�* 

Average reward per step�i��DF�       ��2	�xa�ǩ�A�*

epsilon�i���"�0.       ��W�	^�c�ǩ�A�* 

Average reward per step�i��j�%T       ��2	D�c�ǩ�A�*

epsilon�i����!�.       ��W�	�f�ǩ�A�* 

Average reward per step�i�����J       ��2	Of�ǩ�A�*

epsilon�i����2V.       ��W�	��g�ǩ�A�* 

Average reward per step�i�����G       ��2	��g�ǩ�A�*

epsilon�i��~)�`.       ��W�	-@j�ǩ�A�* 

Average reward per step�i���4�       ��2	�@j�ǩ�A�*

epsilon�i��l���.       ��W�	�k�ǩ�A�* 

Average reward per step�i���k��       ��2	��k�ǩ�A�*

epsilon�i���m�.       ��W�	=�m�ǩ�A�* 

Average reward per step�i����       ��2	��m�ǩ�A�*

epsilon�i����0.       ��W�	�p�ǩ�A�* 

Average reward per step�i��dYVt       ��2	�p�ǩ�A�*

epsilon�i��e�[�.       ��W�	�`r�ǩ�A�* 

Average reward per step�i��t{_�       ��2	(ar�ǩ�A�*

epsilon�i��@��.       ��W�	�t�ǩ�A�* 

Average reward per step�i��f�       ��2	�t�ǩ�A�*

epsilon�i��x�y�.       ��W�	�<v�ǩ�A�* 

Average reward per step�i��kT�       ��2	�=v�ǩ�A�*

epsilon�i��n�n0       ���_	�Xv�ǩ�Ab*#
!
Average reward per episode  ��A�t4.       ��W�	`Yv�ǩ�Ab*!

total reward per episode  ��v>.       ��W�	vz�ǩ�A�* 

Average reward per step  ��E�ڲ       ��2	z�ǩ�A�*

epsilon  �� �].       ��W�	5F|�ǩ�A�* 

Average reward per step  ���-�       ��2	�F|�ǩ�A�*

epsilon  ����`.       ��W�	��~�ǩ�A�* 

Average reward per step  ��F��       ��2	N�~�ǩ�A�*

epsilon  ��x�q#.       ��W�	��ǩ�A�* 

Average reward per step  ��[�?�       ��2	���ǩ�A�*

epsilon  ��.       ��W�	V��ǩ�A�* 

Average reward per step  ��* n       ��2	���ǩ�A�*

epsilon  ���F�.       ��W�	�e��ǩ�A�* 

Average reward per step  ��eT�+       ��2	�f��ǩ�A�*

epsilon  �����.       ��W�	X���ǩ�A�* 

Average reward per step  ���W       ��2	���ǩ�A�*

epsilon  ��ŉ�.       ��W�	����ǩ�A�* 

Average reward per step  ��[�2�       ��2	����ǩ�A�*

epsilon  ��Ҿ��.       ��W�	�ъ�ǩ�A�* 

Average reward per step  ��<��2       ��2	5Ҋ�ǩ�A�*

epsilon  ����[.       ��W�	u��ǩ�A�* 

Average reward per step  ��_"z�       ��2	\��ǩ�A�*

epsilon  ����E�.       ��W�	(,��ǩ�A�* 

Average reward per step  ���"Ψ       ��2	�,��ǩ�A�*

epsilon  ����.       ��W�	 ���ǩ�A�* 

Average reward per step  ������       ��2	����ǩ�A�*

epsilon  ��ɍ�.       ��W�	軒�ǩ�A�* 

Average reward per step  �����       ��2	Ǽ��ǩ�A�*

epsilon  ���c.       ��W�	wה�ǩ�A�* 

Average reward per step  ���:�       ��2	'ؔ�ǩ�A�*

epsilon  ���І�.       ��W�	�J��ǩ�A�* 

Average reward per step  ���l�K       ��2	0K��ǩ�A�*

epsilon  ��iY�|.       ��W�	�n��ǩ�A�* 

Average reward per step  ���"�?       ��2	ro��ǩ�A�*

epsilon  ��a9��.       ��W�	���ǩ�A�* 

Average reward per step  ���h9       ��2	����ǩ�A�*

epsilon  ��q�R.       ��W�	�2��ǩ�A�* 

Average reward per step  ����=       ��2	�3��ǩ�A�*

epsilon  �����o.       ��W�	�i��ǩ�A�* 

Average reward per step  �����H       ��2	�j��ǩ�A�*

epsilon  �����>.       ��W�	E���ǩ�A�* 

Average reward per step  ���*��       ��2	ظ��ǩ�A�*

epsilon  ���S�x.       ��W�	����ǩ�A�* 

Average reward per step  ���`       ��2	*���ǩ�A�*

epsilon  �����1.       ��W�	����ǩ�A�* 

Average reward per step  ��ig}v       ��2	����ǩ�A�*

epsilon  ��|Q`M.       ��W�	���ǩ�A�* 

Average reward per step  ��hm       ��2	���ǩ�A�*

epsilon  ���YQC0       ���_	�7��ǩ�Ac*#
!
Average reward per episode�������.       ��W�	78��ǩ�Ac*!

total reward per episode  �L3E�.       ��W�	k���ǩ�A�* 

Average reward per step���h9m�       ��2	J���ǩ�A�*

epsilon���~g8�.       ��W�	4/��ǩ�A�* 

Average reward per step���ë��       ��2	�/��ǩ�A�*

epsilon���&,�^.       ��W�	E�2�ǩ�A�* 

Average reward per step���wi�       ��2	�2�ǩ�A�*

epsilon���� �.       ��W�	��4�ǩ�A�* 

Average reward per step���ߣF[       ��2	z�4�ǩ�A�*

epsilon����G�.       ��W�	�7�ǩ�A�* 

Average reward per step���� �       ��2	|7�ǩ�A�*

epsilon���J��.       ��W�	^,9�ǩ�A�* 

Average reward per step���:�_Y       ��2	�,9�ǩ�A�*

epsilon�����.       ��W�	�o;�ǩ�A�* 

Average reward per step���\<PF       ��2	ap;�ǩ�A�*

epsilon���z�H.       ��W�	И=�ǩ�A�* 

Average reward per step����<�       ��2	��=�ǩ�A�*

epsilon���-�4.       ��W�	%�?�ǩ�A�* 

Average reward per step����9�       ��2	��?�ǩ�A�*

epsilon���D�!�.       ��W�	�7A�ǩ�A�* 

Average reward per step���0K��       ��2	L8A�ǩ�A�*

epsilon�������.       ��W�	$dC�ǩ�A�* 

Average reward per step���2��       ��2	�dC�ǩ�A�*

epsilon����{|�.       ��W�	��E�ǩ�A�* 

Average reward per step���2QW�       ��2	`�E�ǩ�A�*

epsilon���ˍ?.       ��W�	&�G�ǩ�A�* 

Average reward per step�����2"       ��2	ƧG�ǩ�A�*

epsilon�����lQ.       ��W�	��I�ǩ�A�* 

Average reward per step����MJ       ��2	��I�ǩ�A�*

epsilon�����?�.       ��W�	��K�ǩ�A�* 

Average reward per step���N�1�       ��2	s�K�ǩ�A�*

epsilon���x�v.       ��W�	ϟM�ǩ�A�* 

Average reward per step����F��       ��2	w�M�ǩ�A�*

epsilon�����Ĕ0       ���_	׽M�ǩ�Ad*#
!
Average reward per episode  %����.       ��W�	b�M�ǩ�Ad*!

total reward per episode  %�8�9;.       ��W�	�R�ǩ�A�* 

Average reward per step  %�pO�s       ��2	CR�ǩ�A�*

epsilon  %�3ʷx.       ��W�	?;T�ǩ�A�* 

Average reward per step  %�:!k�       ��2	<T�ǩ�A�*

epsilon  %��^�.       ��W�	��U�ǩ�A�* 

Average reward per step  %��p�g       ��2	l�U�ǩ�A�*

epsilon  %��x��.       ��W�	 W�ǩ�A�* 

Average reward per step  %�i>&�       ��2	� W�ǩ�A�*

epsilon  %��g�.       ��W�	�SY�ǩ�A�* 

Average reward per step  %�"�n       ��2	\TY�ǩ�A�*

epsilon  %��)�v.       ��W�	�e[�ǩ�A�* 

Average reward per step  %�����       ��2	�f[�ǩ�A�*

epsilon  %��<ӣ.       ��W�	�]�ǩ�A�* 

Average reward per step  %�� ;P       ��2	�]�ǩ�A�*

epsilon  %�܅.       ��W�	�	_�ǩ�A�* 

Average reward per step  %�)��+       ��2	�
_�ǩ�A�*

epsilon  %���v.       ��W�	� a�ǩ�A�* 

Average reward per step  %�]C��       ��2	!a�ǩ�A�*

epsilon  %���#.       ��W�	Pc�ǩ�A�* 

Average reward per step  %��Xσ       ��2	Qc�ǩ�A�*

epsilon  %�B���.       ��W�	xe�ǩ�A�* 

Average reward per step  %�z��.       ��2	�xe�ǩ�A�*

epsilon  %�G�J�.       ��W�	ܞg�ǩ�A�* 

Average reward per step  %��_=�       ��2	��g�ǩ�A�*

epsilon  %���Ra.       ��W�	E�i�ǩ�A�* 

Average reward per step  %�؀��       ��2	4�i�ǩ�A�*

epsilon  %���~!.       ��W�	ݕk�ǩ�A�* 

Average reward per step  %���       ��2	��k�ǩ�A�*

epsilon  %�T'!.       ��W�	�m�ǩ�A�* 

Average reward per step  %��&��       ��2	��m�ǩ�A�*

epsilon  %�Ǉ��.       ��W�	;�o�ǩ�A�* 

Average reward per step  %����/       ��2	�o�ǩ�A�*

epsilon  %����.       ��W�	x*r�ǩ�A�* 

Average reward per step  %�gD@�       ��2	$+r�ǩ�A�*

epsilon  %�%"aR.       ��W�	�Xt�ǩ�A�* 

Average reward per step  %���\       ��2	hYt�ǩ�A�*

epsilon  %�����0       ���_	�yt�ǩ�Ae*#
!
Average reward per episode  �e���.       ��W�	hzt�ǩ�Ae*!

total reward per episode  +È��.       ��W�	�.x�ǩ�A�* 

Average reward per step  �P��z       ��2	j/x�ǩ�A�*

epsilon  �}��.       ��W�	�z�ǩ�A�* 

Average reward per step  ��`2       ��2	��z�ǩ�A�*

epsilon  �^�Zi.       ��W�	]�{�ǩ�A�* 

Average reward per step  ��^�       ��2	3�{�ǩ�A�*

epsilon  ���.       ��W�	k+~�ǩ�A�* 

Average reward per step  �M�       ��2	A,~�ǩ�A�*

epsilon  �v�.       ��W�	�Q��ǩ�A�* 

Average reward per step  �+#��       ��2	eR��ǩ�A�*

epsilon  �e^��.       ��W�	�z��ǩ�A�* 

Average reward per step  ��V�"       ��2	�}��ǩ�A�*

epsilon  ����.       ��W�	��ǩ�A�* 

Average reward per step  ��F       ��2	����ǩ�A�*

epsilon  ���.       ��W�	�̆�ǩ�A�* 

Average reward per step  �3�       ��2	F͆�ǩ�A�*

epsilon  ��ल.       ��W�	�<��ǩ�A�* 

Average reward per step  ��I�R       ��2	�=��ǩ�A�*

epsilon  �0k��.       ��W�	���ǩ�A�* 

Average reward per step  �����       ��2	Q��ǩ�A�*

epsilon  �<Y�.       ��W�	5���ǩ�A�* 

Average reward per step  �K��       ��2	���ǩ�A�*

epsilon  �[)��.       ��W�	�Ǝ�ǩ�A�* 

Average reward per step  �>��       ��2	�ǎ�ǩ�A�*

epsilon  �V�ao.       ��W�	8��ǩ�A�* 

Average reward per step  ��&_Y       ��2	���ǩ�A�*

epsilon  �D���.       ��W�	>���ǩ�A�* 

Average reward per step  ��Squ       ��2	ᕒ�ǩ�A�*

epsilon  �*�$s.       ��W�	���ǩ�A�* 

Average reward per step  ��	\       ��2	穔�ǩ�A�*

epsilon  ���.       ��W�	�ʖ�ǩ�A�* 

Average reward per step  ���       ��2	)˖�ǩ�A�*

epsilon  �`1ȳ.       ��W�	A��ǩ�A�* 

Average reward per step  �#��       ��2	��ǩ�A�*

epsilon  ��ޔ0       ���_	�-��ǩ�Af*#
!
Average reward per episode�����I.       ��W�	j.��ǩ�Af*!

total reward per episode   �ߒ0.       ��W�	���ǩ�A�* 

Average reward per step�����m�       ��2	_	��ǩ�A�*

epsilon����I.       ��W�	~5��ǩ�A�* 

Average reward per step���b�u       ��2	&6��ǩ�A�*

epsilon����{�.       ��W�	h��ǩ�A�* 

Average reward per step���[���       ��2	bi��ǩ�A�*

epsilon������.       ��W�	����ǩ�A�* 

Average reward per step���z�       ��2	����ǩ�A�*

epsilon����K�.       ��W�	�-��ǩ�A�* 

Average reward per step���߈��       ��2	o.��ǩ�A�*

epsilon���f��0       ���_	�K��ǩ�Ag*#
!
Average reward per episodeff��"hS.       ��W�	�L��ǩ�Ag*!

total reward per episode  <��M��.       ��W�	���ǩ�A�* 

Average reward per stepff�`_��       ��2	���ǩ�A�*

epsilonff��>�E.       ��W�	7���ǩ�A�* 

Average reward per stepff�t��       ��2	����ǩ�A�*

epsilonff,�.       ��W�	qq��ǩ�A�* 

Average reward per stepffN�$       ��2	Tr��ǩ�A�*

epsilonff�8}.       ��W�	����ǩ�A�* 

Average reward per stepff�        ��2	[���ǩ�A�*

epsilonffv.|.       ��W�	����ǩ�A�* 

Average reward per stepff�^m��       ��2	W���ǩ�A�*

epsilonff)+.       ��W�	5��ǩ�A�* 

Average reward per stepff����y       ��2	���ǩ�A�*

epsilonff��Fw).       ��W�	����ǩ�A�* 

Average reward per stepff����       ��2	Ϻ��ǩ�A�*

epsilonff�$�Er.       ��W�	���ǩ�A�* 

Average reward per stepff���       ��2	����ǩ�A�*

epsilonff��T.       ��W�	�7��ǩ�A�* 

Average reward per stepff��·�       ��2	�8��ǩ�A�*

epsilonff·��9.       ��W�	F���ǩ�A�* 

Average reward per stepff�A<�       ��2	-���ǩ�A�*

epsilonff���.       ��W�	����ǩ�A�* 

Average reward per stepff¨�a�       ��2	I���ǩ�A�*

epsilonff��@n3.       ��W�	�ǩ�A�* 

Average reward per stepff���4       ��2	��ǩ�A�*

epsilonff�9�pa.       ��W�	K!�ǩ�A�* 

Average reward per stepff�ѥ�       ��2	S"�ǩ�A�*

epsilonff� �u�.       ��W�	�C�ǩ�A�* 

Average reward per stepff��~�       ��2	JD�ǩ�A�*

epsilonff»��.       ��W�	�h�ǩ�A�* 

Average reward per stepff��s3"       ��2	�i�ǩ�A�*

epsilonff¡B��.       ��W�	*�
�ǩ�A�* 

Average reward per stepff�l��B       ��2	Ց
�ǩ�A�*

epsilonff���.       ��W�	��ǩ�A�* 

Average reward per stepff���r       ��2	��ǩ�A�*

epsilonffzd�.       ��W�	]P�ǩ�A�* 

Average reward per stepff���9       ��2	Q�ǩ�A�*

epsilonff��yq.       ��W�	Fz�ǩ�A�* 

Average reward per stepff�˿s�       ��2	${�ǩ�A�*

epsilonff��C��.       ��W�	���ǩ�A�* 

Average reward per stepff®��       ��2	���ǩ�A�*

epsilonff�Rrn.       ��W�	�&�ǩ�A�* 

Average reward per stepff�)���       ��2	�'�ǩ�A�*

epsilonff���T.       ��W�	�Z�ǩ�A�* 

Average reward per stepff�1$i+       ��2	�[�ǩ�A�*

epsilonff�p���.       ��W�	�r�ǩ�A�* 

Average reward per stepff��D�       ��2	�s�ǩ�A�*

epsilonff��}�}.       ��W�	���ǩ�A�* 

Average reward per stepff�*�Q       ��2	~��ǩ�A�*

epsilonff�8�.       ��W�	���ǩ�A�* 

Average reward per stepff��8 F       ��2	���ǩ�A�*

epsilonff«�z�.       ��W�	LT�ǩ�A�* 

Average reward per stepff��(6�       ��2	�T�ǩ�A�*

epsilonff�9�W�.       ��W�	�� �ǩ�A�* 

Average reward per stepff�ͯ#       ��2	�� �ǩ�A�*

epsilonff�+�x0.       ��W�	�"�ǩ�A�* 

Average reward per stepff�~�W       ��2	��"�ǩ�A�*

epsilonff���v�.       ��W�	 %�ǩ�A�* 

Average reward per stepff�C#ig       ��2	� %�ǩ�A�*

epsilonff�����.       ��W�	$�&�ǩ�A�* 

Average reward per stepff�$}�       ��2	ܝ&�ǩ�A�*

epsilonff�IQ��.       ��W�	y�(�ǩ�A�* 

Average reward per stepff��v       ��2	W�(�ǩ�A�*

epsilonff���<n.       ��W�	��*�ǩ�A�* 

Average reward per stepff�c0��       ��2	W�*�ǩ�A�*

epsilonff�|A,.       ��W�	F-�ǩ�A�* 

Average reward per stepff�N�       ��2	�F-�ǩ�A�*

epsilonff�o�;D.       ��W�	��.�ǩ�A�* 

Average reward per stepff�l��y       ��2	��.�ǩ�A�*

epsilonff��o:d.       ��W�	��0�ǩ�A�* 

Average reward per stepff�G��3       ��2	8�0�ǩ�A�*

epsilonff�&r.       ��W�	�3�ǩ�A�* 

Average reward per stepff�kU4       ��2	�3�ǩ�A�*

epsilonff�h�#R.       ��W�	r5�ǩ�A�* 

Average reward per stepff��J        ��2	 s5�ǩ�A�*

epsilonff��.       ��W�	s�7�ǩ�A�* 

Average reward per stepff V�       ��2	4�7�ǩ�A�*

epsilonff��4H�.       ��W�	��9�ǩ�A�* 

Average reward per stepff���]       ��2	\�9�ǩ�A�*

epsilonff�ks�l.       ��W�	?n;�ǩ�A�* 

Average reward per stepff�..��       ��2	o;�ǩ�A�*

epsilonff�g��.       ��W�	��=�ǩ�A�* 

Average reward per stepff���[       ��2	P�=�ǩ�A�*

epsilonff�.���.       ��W�	�?�ǩ�A�* 

Average reward per stepff��o�       ��2	��?�ǩ�A�*

epsilonffo��.       ��W�	%YA�ǩ�A�* 

Average reward per stepff���Z       ��2	�YA�ǩ�A�*

epsilonff�;ì.       ��W�	��C�ǩ�A�* 

Average reward per stepff��d�       ��2	f�C�ǩ�A�*

epsilonff�r5��.       ��W�	=�E�ǩ�A�* 

Average reward per stepff��S}       ��2	$�E�ǩ�A�*

epsilonff�{RV�.       ��W�	/�G�ǩ�A�* 

Average reward per stepff�Q��{       ��2	��G�ǩ�A�*

epsilonff�����.       ��W�	�qI�ǩ�A�* 

Average reward per stepff�G<�M       ��2	�rI�ǩ�A�*

epsilonff�8.       ��W�	�(��ǩ�A�* 

Average reward per stepff�)NX�       ��2	5)��ǩ�A�*

epsilonff�h�.       ��W�	���ǩ�A�* 

Average reward per stepff�M9��       ��2	���ǩ�A�*

epsilonff��4�t.       ��W�	Χ��ǩ�A�* 

Average reward per stepff�N��       ��2	����ǩ�A�*

epsilonff���.       ��W�	���ǩ�A�* 

Average reward per stepff�Uv}�       ��2	����ǩ�A�*

epsilonff��m/2.       ��W�	`���ǩ�A�* 

Average reward per stepff�x�a�       ��2	���ǩ�A�*

epsilonff���q.       ��W�	\��ǩ�A�* 

Average reward per stepff��7�        ��2	��ǩ�A�*

epsilonff�mV�.       ��W�	�$��ǩ�A�* 

Average reward per stepff����       ��2	�%��ǩ�A�*

epsilonff�.���.       ��W�	W@��ǩ�A�* 

Average reward per stepff�⬥G       ��2	�@��ǩ�A�*

epsilonff�b�hb.       ��W�	�K��ǩ�A�* 

Average reward per stepff�<�>�       ��2	L��ǩ�A�*

epsilonff�\`�.       ��W�	�V��ǩ�A�* 

Average reward per stepff*<�       ��2	�W��ǩ�A�*

epsilonff�S5M.       ��W�	�o��ǩ�A�* 

Average reward per stepff��?}6       ��2	�p��ǩ�A�*

epsilonff¤��.       ��W�	����ǩ�A�* 

Average reward per stepff>�       ��2	з��ǩ�A�*

epsilonff�°.       ��W�	��ǩ�A�* 

Average reward per stepff��Ҹ�       ��2	���ǩ�A�*

epsilonff�,�(|.       ��W�	s���ǩ�A�* 

Average reward per stepff8X       ��2	���ǩ�A�*

epsilonff����0       ���_		���ǩ�Ah*#
!
Average reward per episodeާ8���&�.       ��W�	����ǩ�Ah*!

total reward per episode  0«s�.       ��W�	)���ǩ�A�* 

Average reward per stepާ8��u        ��2	ꯨ�ǩ�A�*

epsilonާ8����.       ��W�	2ɪ�ǩ�A�* 

Average reward per stepާ8�")$e       ��2	�ɪ�ǩ�A�*

epsilonާ8�N���.       ��W�	����ǩ�A�* 

Average reward per stepާ8�~�       ��2	5��ǩ�A�*

epsilonާ8����.       ��W�	�\��ǩ�A�* 

Average reward per stepާ8�d�O       ��2	�]��ǩ�A�*

epsilonާ8��~I.       ��W�	x��ǩ�A�* 

Average reward per stepާ8�����       ��2	�x��ǩ�A�*

epsilonާ8��7�".       ��W�	����ǩ�A�* 

Average reward per stepާ8�VJ �       ��2	[���ǩ�A�*

epsilonާ8�0Fh.       ��W�	g���ǩ�A�* 

Average reward per stepާ8���f�       ��2	N���ǩ�A�*

epsilonާ8�o��$.       ��W�	�ն�ǩ�A�* 

Average reward per stepާ8�$���       ��2	�ֶ�ǩ�A�*

epsilonާ8��e�.       ��W�	-��ǩ�A�* 

Average reward per stepާ8���       ��2	���ǩ�A�*

epsilonާ8�	�.       ��W�	��ǩ�A�* 

Average reward per stepާ8�	�{�       ��2	���ǩ�A�*

epsilonާ8�b�.       ��W�	(���ǩ�A�* 

Average reward per stepާ8��g;�       ��2	ù��ǩ�A�*

epsilonާ8�ۼ�.       ��W�	+2��ǩ�A�* 

Average reward per stepާ8���J�       ��2	3��ǩ�A�*

epsilonާ8��l��.       ��W�	�O��ǩ�A�* 

Average reward per stepާ8�eӐ�       ��2	eP��ǩ�A�*

epsilonާ8����.       ��W�	����ǩ�A�* 

Average reward per stepާ8���
       ��2	P���ǩ�A�*

epsilonާ8��Nr�.       ��W�	~���ǩ�A�* 

Average reward per stepާ8�Y)�       ��2	K���ǩ�A�*

epsilonާ8��#mG.       ��W�	����ǩ�A�* 

Average reward per stepާ8��:��       ��2	����ǩ�A�*

epsilonާ8�o�.       ��W�	����ǩ�A�* 

Average reward per stepާ8���~       ��2	����ǩ�A�*

epsilonާ8�І��.       ��W�	v��ǩ�A�* 

Average reward per stepާ8�#��[       ��2	H��ǩ�A�*

epsilonާ8��X.       ��W�	�5��ǩ�A�* 

Average reward per stepާ8�s`oX       ��2	�6��ǩ�A�*

epsilonާ8�1sxt.       ��W�	�o��ǩ�A�* 

Average reward per stepާ8�{M%       ��2	�p��ǩ�A�*

epsilonާ8�!c��.       ��W�	;���ǩ�A�* 

Average reward per stepާ8�	��m       ��2	���ǩ�A�*

epsilonާ8����<.       ��W�	n��ǩ�A�* 

Average reward per stepާ8�����       ��2	"��ǩ�A�*

epsilonާ8�E��%.       ��W�	0��ǩ�A�* 

Average reward per stepާ8�U�p       ��2	���ǩ�A�*

epsilonާ8��<h�.       ��W�	@L��ǩ�A�* 

Average reward per stepާ8��?�       ��2	4M��ǩ�A�*

epsilonާ8�]-k.       ��W�	o~��ǩ�A�* 

Average reward per stepާ8��s�       ��2	��ǩ�A�*

epsilonާ8�n��0       ���_	t���ǩ�Ai*#
!
Average reward per episode33���4�n.       ��W�	����ǩ�Ai*!

total reward per episode  ���4�.       ��W�	�k��ǩ�A�* 

Average reward per step33��R�E�       ��2	l��ǩ�A�*

epsilon33���$�.       ��W�	W���ǩ�A�* 

Average reward per step33����e<       ��2	���ǩ�A�*

epsilon33��N��".       ��W�	�k"�ǩ�A�* 

Average reward per step33��YTU       ��2	/l"�ǩ�A�*

epsilon33�����.       ��W�	�%�ǩ�A�* 

Average reward per step33���p&1       ��2	��%�ǩ�A�*

epsilon33��e�/�.       ��W�	�(�ǩ�A�* 

Average reward per step33���-       ��2	�(�ǩ�A�*

epsilon33������.       ��W�	�)�ǩ�A�* 

Average reward per step33��.2~       ��2	��)�ǩ�A�*

epsilon33���v;.       ��W�	��+�ǩ�A�* 

Average reward per step33��P�,2       ��2	:�+�ǩ�A�*

epsilon33���Q�.       ��W�	#�-�ǩ�A�* 

Average reward per step33����h       ��2	��-�ǩ�A�*

epsilon33��DL.       ��W�	E/0�ǩ�A�* 

Average reward per step33��ɴ�       ��2	00�ǩ�A�*

epsilon33���dL.       ��W�	ߩ1�ǩ�A�* 

Average reward per step33��@��       ��2	��1�ǩ�A�*

epsilon33��uG<�.       ��W�	�n4�ǩ�A�* 

Average reward per step33����        ��2	]o4�ǩ�A�*

epsilon33���(�^.       ��W�	2�5�ǩ�A�* 

Average reward per step33���CG�       ��2	��5�ǩ�A�*

epsilon33���2a~.       ��W�	�8�ǩ�A�* 

Average reward per step33���Wb\       ��2	�8�ǩ�A�*

epsilon33���F��.       ��W�	�:�ǩ�A�* 

Average reward per step33���V�       ��2	v:�ǩ�A�*

epsilon33�����.       ��W�	D<�ǩ�A�* 

Average reward per step33���m�       ��2	�D<�ǩ�A�*

epsilon33��z��".       ��W�	��>�ǩ�A�* 

Average reward per step33�����s       ��2	��>�ǩ�A�*

epsilon33���k5�.       ��W�	��@�ǩ�A�* 

Average reward per step33�����2       ��2	)�@�ǩ�A�*

epsilon33����:>0       ���_	��@�ǩ�Aj*#
!
Average reward per episode���!�Uv.       ��W�	�@�ǩ�Aj*!

total reward per episode   �����.       ��W�	�WD�ǩ�A�* 

Average reward per step�������       ��2	�XD�ǩ�A�*

epsilon����}��.       ��W�	��F�ǩ�A�* 

Average reward per step����{       ��2	��F�ǩ�A�*

epsilon����J��.       ��W�	M�H�ǩ�A�* 

Average reward per step������       ��2	"�H�ǩ�A�*

epsilon����Kp�.       ��W�	�^J�ǩ�A�* 

Average reward per step���͹%       ��2	�_J�ǩ�A�*

epsilon����p$.       ��W�	��L�ǩ�A�* 

Average reward per step���^�N�       ��2	y�L�ǩ�A�*

epsilon���}7;.       ��W�	[�N�ǩ�A�* 

Average reward per step���[��       ��2	:�N�ǩ�A�*

epsilon���/��.       ��W�	b�P�ǩ�A�* 

Average reward per step������       ��2	�P�ǩ�A�*

epsilon����Ŷ�.       ��W�	�LS�ǩ�A�* 

Average reward per step�������       ��2	^MS�ǩ�A�*

epsilon�����ƚ.       ��W�	��T�ǩ�A�* 

Average reward per step���[QI&       ��2	b�T�ǩ�A�*

epsilon�������.       ��W�	 �V�ǩ�A�* 

Average reward per step�����-�       ��2	3�V�ǩ�A�*

epsilon���A��[.       ��W�	��X�ǩ�A�* 

Average reward per step������J       ��2	��X�ǩ�A�*

epsilon���ys��.       ��W�	
[�ǩ�A�* 

Average reward per step���� �S       ��2	�[�ǩ�A�*

epsilon���_W�K.       ��W�	`r]�ǩ�A�* 

Average reward per step����j?-       ��2	s]�ǩ�A�*

epsilon�����A�.       ��W�	�_�ǩ�A�* 

Average reward per step���˂�       ��2	{_�ǩ�A�*

epsilon����$��.       ��W�	�Ga�ǩ�A�* 

Average reward per step�����'        ��2	kHa�ǩ�A�*

epsilon����&Q{.       ��W�	z�c�ǩ�A�* 

Average reward per step���^�r       ��2	?�c�ǩ�A�*

epsilon�����G.       ��W�	�!e�ǩ�A�* 

Average reward per step���ϙ��       ��2	�"e�ǩ�A�*

epsilon����/�.       ��W�	�Og�ǩ�A�* 

Average reward per step����.�       ��2	zPg�ǩ�A�*

epsilon���2P��.       ��W�	{ji�ǩ�A�* 

Average reward per step���D^V	       ��2	ki�ǩ�A�*

epsilon���ahbv.       ��W�	`�k�ǩ�A�* 

Average reward per step����ڿ       ��2	p�k�ǩ�A�*

epsilon���}��g.       ��W�	��m�ǩ�A�* 

Average reward per step�����G�       ��2	w�m�ǩ�A�*

epsilon����,.       ��W�	 �o�ǩ�A�* 

Average reward per step����D$�       ��2	��o�ǩ�A�*

epsilon���B�w�0       ���_	p�ǩ�Ak*#
!
Average reward per episode]t��H@�.       ��W�	�p�ǩ�Ak*!

total reward per episode  å\�.       ��W�	t�ǩ�A�* 

Average reward per step]t����-       ��2	�t�ǩ�A�*

epsilon]t��3�E�.       ��W�	>�u�ǩ�A�* 

Average reward per step]t����EN       ��2	ٖu�ǩ�A�*

epsilon]t�����H.       ��W�	��w�ǩ�A�* 

Average reward per step]t��.4e       ��2	g�w�ǩ�A�*

epsilon]t��t� �.       ��W�	r�y�ǩ�A�* 

Average reward per step]t��F_J       ��2	P�y�ǩ�A�*

epsilon]t��;t�:.       ��W�	��{�ǩ�A�* 

Average reward per step]t��As,k       ��2	��{�ǩ�A�*

epsilon]t��Y�P.       ��W�	��}�ǩ�A�* 

Average reward per step]t���=}       ��2	q�}�ǩ�A�*

epsilon]t��ڌ�t.       ��W�	���ǩ�A�* 

Average reward per step]t�����       ��2	���ǩ�A�*

epsilon]t���YN�.       ��W�	eq��ǩ�A�* 

Average reward per step]t���F       ��2	r��ǩ�A�*

epsilon]t���y�.       ��W�	�-��ǩ�A�* 

Average reward per step]t����O�       ��2	�.��ǩ�A�*

epsilon]t���oW(.       ��W�	К��ǩ�A�* 

Average reward per step]t���       ��2	����ǩ�A�*

epsilon]t��\˕;.       ��W�	���ǩ�A�* 

Average reward per step]t���S�L       ��2	s��ǩ�A�*

epsilon]t���E�G.       ��W�	]6��ǩ�A�* 

Average reward per step]t����       ��2	.7��ǩ�A�*

epsilon]t������.       ��W�	4���ǩ�A�* 

Average reward per step]t����       ��2	���ǩ�A�*

epsilon]t���ac.       ��W�	���ǩ�A�* 

Average reward per step]t��J�8�       ��2	���ǩ�A�*

epsilon]t���1��.       ��W�	�>��ǩ�A�* 

Average reward per step]t����:}       ��2	}?��ǩ�A�*

epsilon]t���̄�.       ��W�	.r��ǩ�A�* 

Average reward per step]t�����F       ��2	�r��ǩ�A�*

epsilon]t����o.       ��W�	����ǩ�A�* 

Average reward per step]t��dޣ�       ��2	Y���ǩ�A�*

epsilon]t������.       ��W�	�Ė�ǩ�A�* 

Average reward per step]t��{{*       ��2	�Ŗ�ǩ�A�*

epsilon]t���s�.       ��W�	��ǩ�A�* 

Average reward per step]t���j�       ��2	���ǩ�A�*

epsilon]t��瓕�.       ��W�	�	��ǩ�A�* 

Average reward per step]t��_Bّ       ��2	�
��ǩ�A�*

epsilon]t����5F.       ��W�	����ǩ�A�* 

Average reward per step]t��T�       ��2	U���ǩ�A�*

epsilon]t��1X.       ��W�	;ƞ�ǩ�A�* 

Average reward per step]t���~#�       ��2	�ƞ�ǩ�A�*

epsilon]t��=�p�.       ��W�	���ǩ�A�* 

Average reward per step]t����0l       ��2	���ǩ�A�*

epsilon]t��;��.       ��W�	�?��ǩ�A�* 

Average reward per step]t��.N3       ��2	�@��ǩ�A�*

epsilon]t����)f.       ��W�	.��ǩ�A�* 

Average reward per step]t���Nĝ       ��2	��ǩ�A�*

epsilon]t���l٧.       ��W�	4��ǩ�A�* 

Average reward per step]t��\zt       ��2	5��ǩ�A�*

epsilon]t��3w�:.       ��W�	���ǩ�A�* 

Average reward per step]t��k���       ��2	ﮨ�ǩ�A�*

epsilon]t�����.       ��W�	���ǩ�A�* 

Average reward per step]t���gG       ��2	撫�ǩ�A�*

epsilon]t���-.       ��W�	���ǩ�A�* 

Average reward per step]t�����       ��2	� ��ǩ�A�*

epsilon]t���{�|.       ��W�	�B��ǩ�A�* 

Average reward per step]t����N�       ��2	�C��ǩ�A�*

epsilon]t���3�.       ��W�	4i��ǩ�A�* 

Average reward per step]t��7��       ��2	�i��ǩ�A�*

epsilon]t��R��.       ��W�	<���ǩ�A�* 

Average reward per step]t��k�k       ��2	'���ǩ�A�*

epsilon]t���s&.       ��W�	zƷ�ǩ�A�* 

Average reward per step]t���Մ�       ��2	PǷ�ǩ�A�*

epsilon]t��ܱ��.       ��W�	���ǩ�A�* 

Average reward per step]t���B�       ��2	�	��ǩ�A�*

epsilon]t��V��.       ��W�	ʍ��ǩ�A�* 

Average reward per step]t��=S��       ��2	����ǩ�A�*

epsilon]t���g�.       ��W�	����ǩ�A�* 

Average reward per step]t���GD!       ��2	o���ǩ�A�*

epsilon]t��L��.       ��W�	���ǩ�A�* 

Average reward per step]t����       ��2	����ǩ�A�*

epsilon]t��>[wZ.       ��W�	#���ǩ�A�* 

Average reward per step]t���Y��       ��2	���ǩ�A�*

epsilon]t��`�
�.       ��W�	���ǩ�A�* 

Average reward per step]t��H�ye       ��2	����ǩ�A�*

epsilon]t�����.       ��W�	w���ǩ�A�* 

Average reward per step]t��a�V       ��2	���ǩ�A�*

epsilon]t���`�.       ��W�	#.��ǩ�A�* 

Average reward per step]t���       ��2	�.��ǩ�A�*

epsilon]t���F$3.       ��W�	����ǩ�A�* 

Average reward per step]t���1��       ��2	����ǩ�A�*

epsilon]t�����.       ��W�	�}��ǩ�A�* 

Average reward per step]t��t��       ��2	�~��ǩ�A�*

epsilon]t�����.       ��W�	���ǩ�A�* 

Average reward per step]t���7       ��2	���ǩ�A�*

epsilon]t���.       ��W�	C:��ǩ�A�* 

Average reward per step]t���eH�       ��2	;��ǩ�A�*

epsilon]t����=�0       ���_	O[��ǩ�Al*#
!
Average reward per episode����v�'.       ��W�	�[��ǩ�Al*!

total reward per episode  ���L.       ��W�	 ��ǩ�A�* 

Average reward per step�����g       ��2	� ��ǩ�A�*

epsilon���0I�7.       ��W�	I��ǩ�A�* 

Average reward per step���#��       ��2	�I��ǩ�A�*

epsilon���yQ^�.       ��W�	Pr��ǩ�A�* 

Average reward per step����       ��2	!s��ǩ�A�*

epsilon���X�4�.       ��W�	v���ǩ�A�* 

Average reward per step�������       ��2	a���ǩ�A�*

epsilon���Y�%.       ��W�	�;��ǩ�A�* 

Average reward per step����X��       ��2	�<��ǩ�A�*

epsilon����"�.       ��W�	�w��ǩ�A�* 

Average reward per step����<L^       ��2	y��ǩ�A�*

epsilon���G�U�.       ��W�	u���ǩ�A�* 

Average reward per step���>�+       ��2	`���ǩ�A�*

epsilon����$d .       ��W�	���ǩ�A�* 

Average reward per step����w�       ��2	���ǩ�A�*

epsilon��� ���.       ��W�	4���ǩ�A�* 

Average reward per step������       ��2	׽��ǩ�A�*

epsilon�������.       ��W�	����ǩ�A�* 

Average reward per step���i�       ��2	����ǩ�A�*

epsilon���#���.       ��W�	%��ǩ�A�* 

Average reward per step�����       ��2	�%��ǩ�A�*

epsilon������e.       ��W�	HR��ǩ�A�* 

Average reward per step���^"       ��2	vS��ǩ�A�*

epsilon���m��.       ��W�	L���ǩ�A�* 

Average reward per step���,�&�       ��2	*���ǩ�A�*

epsilon���,9��.       ��W�	�/��ǩ�A�* 

Average reward per step����S��       ��2	�0��ǩ�A�*

epsilon����߇�.       ��W�	p^��ǩ�A�* 

Average reward per step���lI�       ��2	B_��ǩ�A�*

epsilon������.       ��W�	f���ǩ�A�* 

Average reward per step�����       ��2	����ǩ�A�*

epsilon�����e�.       ��W�	S��ǩ�A�* 

Average reward per step���m��       ��2	�S��ǩ�A�*

epsilon���ƙ�.       ��W�	V���ǩ�A�* 

Average reward per step�����Ub       ��2	,���ǩ�A�*

epsilon���[�{�.       ��W�	m���ǩ�A�* 

Average reward per step���
���       ��2	T���ǩ�A�*

epsilon����}.       ��W�	���ǩ�A�* 

Average reward per step�����M       ��2	d��ǩ�A�*

epsilon����4�.       ��W�	�S��ǩ�A�* 

Average reward per step���]�?       ��2	7T��ǩ�A�*

epsilon����0��.       ��W�	3��ǩ�A�* 

Average reward per step���B���       ��2	��ǩ�A�*

epsilon����qv.       ��W�	���ǩ�A�* 

Average reward per step���4�Q       ��2	���ǩ�A�*

epsilon���eP�.       ��W�	S�ǩ�A�* 

Average reward per step���T:6�       ��2	�S�ǩ�A�*

epsilon����M��0       ���_	;q�ǩ�Am*#
!
Average reward per episode  ���W��.       ��W�	�q�ǩ�Am*!

total reward per episode  �n��.       ��W�	`�	�ǩ�A�* 

Average reward per step  ��g̲o       ��2	 �	�ǩ�A�*

epsilon  ���݋g.       ��W�	w�ǩ�A�* 

Average reward per step  ������       ��2	8�ǩ�A�*

epsilon  ���:�.       ��W�	��ǩ�A�* 

Average reward per step  ��bvS�       ��2	6�ǩ�A�*

epsilon  ���IH�.       ��W�	��ǩ�A�* 

Average reward per step  ��\G��       ��2	��ǩ�A�*

epsilon  ���5�.       ��W�	���ǩ�A�* 

Average reward per step  ��^3%B       ��2	���ǩ�A�*

epsilon  ����.       ��W�	[�ǩ�A�* 

Average reward per step  ����LA       ��2	W�ǩ�A�*

epsilon  ���	y�.       ��W�	̴�ǩ�A�* 

Average reward per step  ����(4       ��2	���ǩ�A�*

epsilon  ��Ԓ�.       ��W�	-��ǩ�A�* 

Average reward per step  ������       ��2	��ǩ�A�*

epsilon  ���=�.       ��W�	�$�ǩ�A�* 

Average reward per step  ����       ��2	�%�ǩ�A�*

epsilon  ��<�.       ��W�	~R�ǩ�A�* 

Average reward per step  ���'�       ��2	;S�ǩ�A�*

epsilon  ����bM.       ��W�	���ǩ�A�* 

Average reward per step  ���><       ��2	���ǩ�A�*

epsilon  ��%���.       ��W�	3 �ǩ�A�* 

Average reward per step  ���"�J       ��2	� �ǩ�A�*

epsilon  ���?��.       ��W�	=�"�ǩ�A�* 

Average reward per step  �����       ��2	,�"�ǩ�A�*

epsilon  ��CA�.       ��W�	�p&�ǩ�A�* 

Average reward per step  ��!���       ��2	�q&�ǩ�A�*

epsilon  ���@L.       ��W�	v�(�ǩ�A�* 

Average reward per step  ���}�w       ��2	"�(�ǩ�A�*

epsilon  ��|Y�.       ��W�	�6*�ǩ�A�* 

Average reward per step  ��V5`�       ��2	�7*�ǩ�A�*

epsilon  ������.       ��W�	�Q,�ǩ�A�* 

Average reward per step  ���*��       ��2	iR,�ǩ�A�*

epsilon  ��r�X0       ���_	�n,�ǩ�An*#
!
Average reward per episodeKK�?��.       ��W�	�o,�ǩ�An*!

total reward per episode  Ï#��.       ��W�	;�0�ǩ�A�* 

Average reward per stepKK� ��f       ��2	��0�ǩ�A�*

epsilonKK�f�i.       ��W�	�3�ǩ�A�* 

Average reward per stepKK��`}�       ��2	W3�ǩ�A�*

epsilonKK����.       ��W�	�4�ǩ�A�* 

Average reward per stepKK��M)�       ��2	�4�ǩ�A�*

epsilonKK�|\O�.       ��W�	��6�ǩ�A�* 

Average reward per stepKK��K��       ��2	C�6�ǩ�A�*

epsilonKK��y�.       ��W�	�8�ǩ�A�* 

Average reward per stepKK����,       ��2	��8�ǩ�A�*

epsilonKK�dj.       ��W�	�1;�ǩ�A�* 

Average reward per stepKK�7U$�       ��2	+3;�ǩ�A�*

epsilonKK��v�.       ��W�	��<�ǩ�A�* 

Average reward per stepKK���}       ��2	L�<�ǩ�A�*

epsilonKK��e�.       ��W�	�?�ǩ�A�* 

Average reward per stepKK����+       ��2	??�ǩ�A�*

epsilonKK����.       ��W�	NGA�ǩ�A�* 

Average reward per stepKK��h��       ��2	�GA�ǩ�A�*

epsilonKK��u[?.       ��W�	vkC�ǩ�A�* 

Average reward per stepKK�����       ��2	HlC�ǩ�A�*

epsilonKK���
�.       ��W�	$�D�ǩ�A�* 

Average reward per stepKK���       ��2	�D�ǩ�A�*

epsilonKK�z��7.       ��W�	. H�ǩ�A�* 

Average reward per stepKK��*       ��2	� H�ǩ�A�*

epsilonKK��W��.       ��W�	�K�ǩ�A�* 

Average reward per stepKK�"v�       ��2	̖K�ǩ�A�*

epsilonKK�C:fX.       ��W�	x�M�ǩ�A�* 

Average reward per stepKK��S�       ��2	عM�ǩ�A�*

epsilonKK��+u�.       ��W�	�`O�ǩ�A�* 

Average reward per stepKK�sGF�       ��2	�aO�ǩ�A�*

epsilonKK�dC9q.       ��W�	�~Q�ǩ�A�* 

Average reward per stepKK��(m6       ��2	�Q�ǩ�A�*

epsilonKK�XR��.       ��W�	�S�ǩ�A�* 

Average reward per stepKK��)       ��2	��S�ǩ�A�*

epsilonKK��N��.       ��W�	��U�ǩ�A�* 

Average reward per stepKK��eoT       ��2	W�U�ǩ�A�*

epsilonKK���ȑ0       ���_	B'V�ǩ�Ao*#
!
Average reward per episodeUU���p�.       ��W�	�'V�ǩ�Ao*!

total reward per episode  �׋�|.       ��W�	C�Y�ǩ�A�* 

Average reward per stepUU�#��       ��2	 �Y�ǩ�A�*

epsilonUU���X�.       ��W�	[\�ǩ�A�* 

Average reward per stepUU����       ��2	[\�ǩ�A�*

epsilonUU��_�S.       ��W�	^�ǩ�A�* 

Average reward per stepUU�N��       ��2	^�ǩ�A�*

epsilonUU���t.       ��W�	/`�ǩ�A�* 

Average reward per stepUU��>`c       ��2	�/`�ǩ�A�*

epsilonUU����.       ��W�	�;b�ǩ�A�* 

Average reward per stepUU�ɤ��       ��2	><b�ǩ�A�*

epsilonUU��.�.       ��W�	_�c�ǩ�A�* 

Average reward per stepUU���|       ��2	��c�ǩ�A�*

epsilonUU�b�.       ��W�	��e�ǩ�A�* 

Average reward per stepUU�답�       ��2	)�e�ǩ�A�*

epsilonUU�Vމ�.       ��W�	 h�ǩ�A�* 

Average reward per stepUU���       ��2	� h�ǩ�A�*

epsilonUU�"N.       ��W�	�3j�ǩ�A�* 

Average reward per stepUU�1k�l       ��2	�4j�ǩ�A�*

epsilonUU���^.       ��W�	fl�ǩ�A�* 

Average reward per stepUU��0�B       ��2	�fl�ǩ�A�*

epsilonUU����.       ��W�	��n�ǩ�A�* 

Average reward per stepUU���       ��2	��n�ǩ�A�*

epsilonUU�bD�.       ��W�	��p�ǩ�A�* 

Average reward per stepUU��.[�       ��2	i�p�ǩ�A�*

epsilonUU��(�.       ��W�	Dir�ǩ�A�* 

Average reward per stepUU�m��@       ��2	�ir�ǩ�A�*

epsilonUU��-�.       ��W�	��t�ǩ�A�* 

Average reward per stepUU���       ��2	Ϣt�ǩ�A�*

epsilonUU��Z�.       ��W�	)�v�ǩ�A�* 

Average reward per stepUU�O[�       ��2	�v�ǩ�A�*

epsilonUU��0�.       ��W�	\Tx�ǩ�A�* 

Average reward per stepUU���o�       ��2	;Ux�ǩ�A�*

epsilonUU�Ɯ�N.       ��W�	�yz�ǩ�A�* 

Average reward per stepUU�h>=X       ��2	pzz�ǩ�A�*

epsilonUU��<�R.       ��W�	x�|�ǩ�A�* 

Average reward per stepUU��&
-       ��2	��|�ǩ�A�*

epsilonUU�U�%.       ��W�	)�ǩ�A�* 

Average reward per stepUU�\��+       ��2	��ǩ�A�*

epsilonUU��r8m.       ��W�	g*��ǩ�A�* 

Average reward per stepUU�^6�       ��2	=+��ǩ�A�*

epsilonUU��Z
.       ��W�	<N��ǩ�A�* 

Average reward per stepUU���Q�       ��2		O��ǩ�A�*

epsilonUU�:JP�.       ��W�	����ǩ�A�* 

Average reward per stepUU��~       ��2	���ǩ�A�*

epsilonUU�I�"�.       ��W�	����ǩ�A�* 

Average reward per stepUU�x        ��2	����ǩ�A�*

epsilonUU��{E�.       ��W�	����ǩ�A�* 

Average reward per stepUU�n}�1       ��2	����ǩ�A�*

epsilonUU��(��.       ��W�	k��ǩ�A�* 

Average reward per stepUU���h�       ��2	���ǩ�A�*

epsilonUU��u-.       ��W�	!��ǩ�A�* 

Average reward per stepUU��;�       ��2	"��ǩ�A�*

epsilonUU���.       ��W�	1��ǩ�A�* 

Average reward per stepUU��[ζ       ��2	�1��ǩ�A�*

epsilonUU�iaӄ.       ��W�	�[��ǩ�A�* 

Average reward per stepUU��
�H       ��2	l\��ǩ�A�*

epsilonUU��i �.       ��W�	�k��ǩ�A�* 

Average reward per stepUU��{{�       ��2	rl��ǩ�A�*

epsilonUU���f�.       ��W�	�x��ǩ�A�* 

Average reward per stepUU�j��       ��2	�y��ǩ�A�*

epsilonUU�>0�.       ��W�		���ǩ�A�* 

Average reward per stepUU�ĭzv       ��2	֩��ǩ�A�*

epsilonUU�͒�.       ��W�	�H��ǩ�A�* 

Average reward per stepUU���\       ��2	oI��ǩ�A�*

epsilonUU�,.�\0       ���_	`��ǩ�Ap*#
!
Average reward per episode  r�M1'.       ��W�	�`��ǩ�Ap*!

total reward per episode  ��jI��.       ��W�	k��ǩ�A�* 

Average reward per step  r�;?��       ��2	b��ǩ�A�*

epsilon  r���_(.       ��W�	����ǩ�A�* 

Average reward per step  r�A�z       ��2	����ǩ�A�*

epsilon  r��8� .       ��W�	F���ǩ�A�* 

Average reward per step  r�3       ��2	���ǩ�A�*

epsilon  r���.       ��W�	����ǩ�A�* 

Average reward per step  r��֗       ��2	5���ǩ�A�*

epsilon  r�T.�3.       ��W�	�r��ǩ�A�* 

Average reward per step  r�fX��       ��2	.s��ǩ�A�*

epsilon  r�����.       ��W�	����ǩ�A�* 

Average reward per step  r�����       ��2	~���ǩ�A�*

epsilon  r����.       ��W�	���ǩ�A�* 

Average reward per step  r�2	��       ��2	����ǩ�A�*

epsilon  r�".       ��W�	����ǩ�A�* 

Average reward per step  r��L�       ��2	����ǩ�A�*

epsilon  r��l#�.       ��W�	q���ǩ�A�* 

Average reward per step  r���=       ��2	)���ǩ�A�*

epsilon  r��0�@.       ��W�	:���ǩ�A�* 

Average reward per step  r�*7��       ��2	����ǩ�A�*

epsilon  r���1.       ��W�	���ǩ�A�* 

Average reward per step  r�3�        ��2	����ǩ�A�*

epsilon  r�Tj2.       ��W�	�M��ǩ�A�* 

Average reward per step  r��-X       ��2	jN��ǩ�A�*

epsilon  r��.o�.       ��W�	����ǩ�A�* 

Average reward per step  r����       ��2	W���ǩ�A�*

epsilon  r�K�gr.       ��W�	����ǩ�A�* 

Average reward per step  r��!�c       ��2	����ǩ�A�*

epsilon  r�/i�r.       ��W�	���ǩ�A�* 

Average reward per step  r�Q�O       ��2	���ǩ�A�*

epsilon  r����C.       ��W�	cB��ǩ�A�* 

Average reward per step  r�e��{       ��2	C��ǩ�A�*

epsilon  r�<���.       ��W�	���ǩ�A�* 

Average reward per step  r�^W��       ��2		���ǩ�A�*

epsilon  r�?�b�.       ��W�	���ǩ�A�* 

Average reward per step  r��:)�       ��2	����ǩ�A�*

epsilon  r�r�2.       ��W�	��ǩ�A�* 

Average reward per step  r���4=       ��2	��ǩ�A�*

epsilon  r�+q�.       ��W�	�D�ǩ�A�* 

Average reward per step  r����g       ��2	�E�ǩ�A�*

epsilon  r�E�1�.       ��W�	|��ǩ�A�* 

Average reward per step  r�&�i�       ��2	k��ǩ�A�*

epsilon  r�(��M.       ��W�	��ǩ�A�* 

Average reward per step  r�j �)       ��2	��ǩ�A�*

epsilon  r��F�0       ���_	���ǩ�Aq*#
!
Average reward per episode  ��	q@.       ��W�	`��ǩ�Aq*!

total reward per episode  �|1fv.       ��W�	eP�ǩ�A�* 

Average reward per step  �����b       ��2	Q�ǩ�A�*

epsilon  ��C�^�.       ��W�	�a�ǩ�A�* 

Average reward per step  ��o�]z       ��2	Rb�ǩ�A�*

epsilon  ��Q C�.       ��W�	�m�ǩ�A�* 

Average reward per step  �����       ��2	Ln�ǩ�A�*

epsilon  ���zp�.       ��W�	N��ǩ�A�* 

Average reward per step  ��Y6�       ��2	��ǩ�A�*

epsilon  ��)T��.       ��W�	\��ǩ�A�* 

Average reward per step  ��r7��       ��2	2��ǩ�A�*

epsilon  ����C[.       ��W�	���ǩ�A�* 

Average reward per step  ���jE       ��2	g��ǩ�A�*

epsilon  ���z7.       ��W�	7��ǩ�A�* 

Average reward per step  ��:6��       ��2	���ǩ�A�*

epsilon  ��<,�.       ��W�	��ǩ�A�* 

Average reward per step  ��Ha�m       ��2	p	�ǩ�A�*

epsilon  ��T�q�.       ��W�	���ǩ�A�* 

Average reward per step  ���_��       ��2	w��ǩ�A�*

epsilon  ��Wf�2.       ��W�	]��ǩ�A�* 

Average reward per step  ��
���       ��2	��ǩ�A�*

epsilon  ����۲.       ��W�	���ǩ�A�* 

Average reward per step  ��!�J�       ��2	#��ǩ�A�*

epsilon  ���W�Y.       ��W�	��!�ǩ�A�* 

Average reward per step  ����E       ��2	4�!�ǩ�A�*

epsilon  ��S?>�.       ��W�	r6$�ǩ�A�* 

Average reward per step  ��ﱅ�       ��2	?7$�ǩ�A�*

epsilon  ��c��.       ��W�	8�%�ǩ�A�* 

Average reward per step  ��߃O�       ��2	�%�ǩ�A�*

epsilon  ����.       ��W�	}�'�ǩ�A�* 

Average reward per step  ��- �       ��2	K�'�ǩ�A�*

epsilon  ����.       ��W�	O*�ǩ�A�* 

Average reward per step  ��qMYv       ��2	�O*�ǩ�A�*

epsilon  �����0       ���_	�l*�ǩ�Ar*#
!
Average reward per episode  )����.       ��W�	zm*�ǩ�Ar*!

total reward per episode  )ó�1.       ��W�	 �-�ǩ�A�* 

Average reward per step  )���       ��2	��-�ǩ�A�*

epsilon  )�Z��.       ��W�	�*0�ǩ�A�* 

Average reward per step  )�L�y       ��2	k+0�ǩ�A�*

epsilon  )���&.       ��W�	�^2�ǩ�A�* 

Average reward per step  )��0       ��2	c_2�ǩ�A�*

epsilon  )���J.       ��W�	|�4�ǩ�A�* 

Average reward per step  )�{l��       ��2	I�4�ǩ�A�*

epsilon  )�f�_�.       ��W�	ҩ6�ǩ�A�* 

Average reward per step  )��}�O       ��2	z�6�ǩ�A�*

epsilon  )���d�.       ��W�	��8�ǩ�A�* 

Average reward per step  )�~It�       ��2	��8�ǩ�A�*

epsilon  )��D�.       ��W�	X�|�ǩ�A�* 

Average reward per step  )�ﻜ�       ��2	��|�ǩ�A�*

epsilon  )�� T.       ��W�	�ǩ�A�* 

Average reward per step  )��ڋQ       ��2	��ǩ�A�*

epsilon  )�����.       ��W�	����ǩ�A�* 

Average reward per step  )�����       ��2	����ǩ�A�*

epsilon  )���.       ��W�	\8��ǩ�A�* 

Average reward per step  )����       ��2	 9��ǩ�A�*

epsilon  )�=dȲ.       ��W�	�J��ǩ�A�* 

Average reward per step  )���       ��2	
L��ǩ�A�*

epsilon  )�,j�6.       ��W�	�r��ǩ�A�* 

Average reward per step  )�r�       ��2	�s��ǩ�A�*

epsilon  )��ڔ.       ��W�	����ǩ�A�* 

Average reward per step  )���w/       ��2	}���ǩ�A�*

epsilon  )��w�.       ��W�	�ʋ�ǩ�A�* 

Average reward per step  )�o��       ��2	>ˋ�ǩ�A�*

epsilon  )�!CH.       ��W�	�R��ǩ�A�* 

Average reward per step  )����       ��2	TS��ǩ�A�*

epsilon  )�8j��.       ��W�	Yi��ǩ�A�* 

Average reward per step  )��F-2       ��2	�i��ǩ�A�*

epsilon  )�"� �.       ��W�	���ǩ�A�* 

Average reward per step  )����       ��2	ō��ǩ�A�*

epsilon  )��Z�A.       ��W�	���ǩ�A�* 

Average reward per step  )�R�u       ��2	맓�ǩ�A�*

epsilon  )�z�G�.       ��W�	t��ǩ�A�* 

Average reward per step  )�˘05       ��2	c��ǩ�A�*

epsilon  )�i��.       ��W�	+k��ǩ�A�* 

Average reward per step  )��T|5       ��2	/l��ǩ�A�*

epsilon  )�&Ju�.       ��W�	�ʙ�ǩ�A�* 

Average reward per step  )�%_U?       ��2	�˙�ǩ�A�*

epsilon  )�	`��.       ��W�	G��ǩ�A�* 

Average reward per step  )��B�#       ��2	���ǩ�A�*

epsilon  )�p�p.       ��W�	���ǩ�A�* 

Average reward per step  )�u��       ��2	���ǩ�A�*

epsilon  )�5�M.       ��W�	�#��ǩ�A�* 

Average reward per step  )���       ��2	p$��ǩ�A�*

epsilon  )���'�.       ��W�	Zg��ǩ�A�* 

Average reward per step  )�͂v       ��2	<h��ǩ�A�*

epsilon  )�.W��.       ��W�	Mۣ�ǩ�A�* 

Average reward per step  )�����       ��2	�ۣ�ǩ�A�*

epsilon  )��-�.       ��W�	�(��ǩ�A�* 

Average reward per step  )��q?�       ��2	$)��ǩ�A�*

epsilon  )�4-��.       ��W�	�L��ǩ�A�* 

Average reward per step  )�� 
       ��2	�M��ǩ�A�*

epsilon  )�x�j{.       ��W�	����ǩ�A�* 

Average reward per step  )�\�d�       ��2	����ǩ�A�*

epsilon  )�L��"0       ���_	*���ǩ�As*#
!
Average reward per episode�i��;F�.       ��W�	窱�ǩ�As*!

total reward per episode  ����.       ��W�	|���ǩ�A�* 

Average reward per step�i�����;       ��2	g���ǩ�A�*

epsilon�i��edۻ.       ��W�	���ǩ�A�* 

Average reward per step�i��JY �       ��2	x��ǩ�A�*

epsilon�i��^��.       ��W�	�p��ǩ�A�* 

Average reward per step�i��
���       ��2	�q��ǩ�A�*

epsilon�i���>�.       ��W�	�t��ǩ�A�* 

Average reward per step�i������       ��2	�u��ǩ�A�*

epsilon�i��v��N.       ��W�	����ǩ�A�* 

Average reward per step�i����҅       ��2	i���ǩ�A�*

epsilon�i��Z`��.       ��W�	Ov��ǩ�A�* 

Average reward per step�i��h�=�       ��2	6w��ǩ�A�*

epsilon�i��ڷ�.       ��W�	4���ǩ�A�* 

Average reward per step�i����h       ��2	���ǩ�A�*

epsilon�i���s�.       ��W�	����ǩ�A�* 

Average reward per step�i��0g��       ��2	)���ǩ�A�*

epsilon�i���=�w.       ��W�	���ǩ�A�* 

Average reward per step�i��ks�       ��2	n��ǩ�A�*

epsilon�i���3��.       ��W�	?R��ǩ�A�* 

Average reward per step�i���[       ��2	TS��ǩ�A�*

epsilon�i���{��.       ��W�	; ��ǩ�A�* 

Average reward per step�i��x�,�       ��2	� ��ǩ�A�*

epsilon�i�����.       ��W�	�+��ǩ�A�* 

Average reward per step�i��- f&       ��2	�,��ǩ�A�*

epsilon�i���/�I.       ��W�	�`��ǩ�A�* 

Average reward per step�i���        ��2	ka��ǩ�A�*

epsilon�i��%X�.       ��W�	����ǩ�A�* 

Average reward per step�i��ժu~       ��2	>���ǩ�A�*

epsilon�i��~sq�.       ��W�	a2��ǩ�A�* 

Average reward per step�i���m�       ��2	3��ǩ�A�*

epsilon�i����(@.       ��W�	T��ǩ�A�* 

Average reward per step�i��9�       ��2	&U��ǩ�A�*

epsilon�i���@��.       ��W�	�z�