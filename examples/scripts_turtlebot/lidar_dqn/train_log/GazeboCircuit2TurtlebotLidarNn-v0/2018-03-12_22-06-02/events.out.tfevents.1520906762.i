       �K"	  ��̩�Abrain.Event:2V�\5��      �*	����̩�A"��
z
flatten_1_inputPlaceholder*+
_output_shapes
:���������* 
shape:���������*
dtype0
^
flatten_1/ShapeShapeflatten_1_input*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
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
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*0
_output_shapes
:������������������*
T0*
Tshape0
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
seed���)*
T0*
dtype0*
_output_shapes
:	�*
seed2���
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
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
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(
|
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes
:	�*
T0*!
_class
loc:@dense_1/kernel
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:�*
T0*
_class
loc:@dense_1/bias
�
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
_output_shapes
:	�d*
seed2���*
seed���)*
T0*
dtype0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	�d*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�d
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
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
_output_shapes
:	�d*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
|
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�d
Z
dense_2/ConstConst*
_output_shapes
:d*
valueBd*    *
dtype0
x
dense_2/bias
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:d
�
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������d*
T0
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
dense_3/random_uniform/minConst*
valueB
 *��L�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2�ځ*
seed���)
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
VariableV2*
shared_name *
dtype0*
_output_shapes

:d2*
	container *
shape
:d2
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
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:2*
use_locking(
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:2
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
activation_3/ReluReludense_3/BiasAdd*
T0*'
_output_shapes
:���������2
m
dense_4/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"2      
_
dense_4/random_uniform/minConst*
valueB
 *�D��*
dtype0*
_output_shapes
: 
_
dense_4/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�D�>*
dtype0
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
_output_shapes

:2*
seed2��K*
seed���)*
T0*
dtype0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
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
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:2*
use_locking(
{
dense_4/kernel/readIdentitydense_4/kernel*
_output_shapes

:2*
T0*!
_class
loc:@dense_4/kernel
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
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(
q
dense_4/bias/readIdentitydense_4/bias*
T0*
_class
loc:@dense_4/bias*
_output_shapes
:
�
dense_4/MatMulMatMulactivation_3/Reludense_4/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
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
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(
{
dense_5/kernel/readIdentitydense_5/kernel*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
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
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:
q
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
_output_shapes
:*
T0
�
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
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
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
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
dtype0*
_output_shapes
:*
valueB"      
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
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������
_
lambda_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*
_output_shapes

:*

Tidx0*
	keep_dims(*
T0
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
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0	
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
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
valueB
 *o�9*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
^
Adam/beta_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
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
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
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
Adam/decayAdam/decay/initial_value*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: *
use_locking(
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
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
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
[
flatten_1_1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
^
flatten_1_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*

axis *
N*
_output_shapes
:*
T0
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
seed���)*
T0*
dtype0*
_output_shapes
:	�*
seed2��h
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
VariableV2*
shape:	�*
shared_name *
dtype0*
_output_shapes
:	�*
	container 
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
dense_2_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"   d   *
dtype0
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
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
dtype0*
_output_shapes
:	�d*
seed2Ӽ�*
seed���)*
T0
�
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
_output_shapes
:	�d*
T0
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
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d
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
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@dense_2_1/bias
w
dense_2_1/bias/readIdentitydense_2_1/bias*
T0*!
_class
loc:@dense_2_1/bias*
_output_shapes
:d
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
�
dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
`
activation_2_1/ReluReludense_2_1/BiasAdd*
T0*'
_output_shapes
:���������d
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
dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
_output_shapes

:d2*
T0
�
dense_3_1/kernel
VariableV2*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name *
dtype0
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
VariableV2*
_output_shapes
:2*
	container *
shape:2*
shared_name *
dtype0
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
dense_3_1/bias/readIdentitydense_3_1/bias*!
_class
loc:@dense_3_1/bias*
_output_shapes
:2*
T0
�
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*
T0*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( 
�
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������2*
T0
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
seed���)*
T0*
dtype0*
_output_shapes

:2*
seed2ֹM
�
dense_4_1/random_uniform/subSubdense_4_1/random_uniform/maxdense_4_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_4_1/random_uniform/mulMul&dense_4_1/random_uniform/RandomUniformdense_4_1/random_uniform/sub*
_output_shapes

:2*
T0
�
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
_output_shapes

:2*
T0
�
dense_4_1/kernel
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
shared_name *
dtype0
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
dense_4_1/bias/readIdentitydense_4_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_4_1/bias
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
dense_5_1/random_uniform/maxConst*
valueB
 *�m?*
dtype0*
_output_shapes
: 
�
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:*
seed2��
�
dense_5_1/random_uniform/subSubdense_5_1/random_uniform/maxdense_5_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_5_1/random_uniform/mulMul&dense_5_1/random_uniform/RandomUniformdense_5_1/random_uniform/sub*
_output_shapes

:*
T0
�
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
T0*
_output_shapes

:
�
dense_5_1/kernel
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
dense_5_1/bias/readIdentitydense_5_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_5_1/bias
�
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:���������
d
lambda_1_1/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
q
 lambda_1_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 
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
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
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
valueB"       *
dtype0*
_output_shapes
:
�
lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*
_output_shapes

:*

Tidx0*
	keep_dims(*
T0
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*
T0*'
_output_shapes
:���������
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_4/kernel
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
IsVariableInitialized_9IsVariableInitializeddense_5/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_5/bias
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
IsVariableInitialized_16IsVariableInitializeddense_1_1/bias*!
_class
loc:@dense_1_1/bias*
dtype0*
_output_shapes
: 
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
AssignAssigndense_1_1/kernelPlaceholder*
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking( 
X
Placeholder_1Placeholder*
dtype0*
_output_shapes	
:�*
shape:�
�
Assign_1Assigndense_1_1/biasPlaceholder_1*
use_locking( *
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:�
`
Placeholder_2Placeholder*
shape:	�d*
dtype0*
_output_shapes
:	�d
�
Assign_2Assigndense_2_1/kernelPlaceholder_2*
use_locking( *
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
:d*
shape:d
�
Assign_3Assigndense_2_1/biasPlaceholder_3*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d*
use_locking( 
^
Placeholder_4Placeholder*
shape
:d2*
dtype0*
_output_shapes

:d2
�
Assign_4Assigndense_3_1/kernelPlaceholder_4*
validate_shape(*
_output_shapes

:d2*
use_locking( *
T0*#
_class
loc:@dense_3_1/kernel
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
:2*
shape:2
�
Assign_5Assigndense_3_1/biasPlaceholder_5*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2*
use_locking( *
T0
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
Placeholder_8Placeholder*
dtype0*
_output_shapes

:*
shape
:
�
Assign_8Assigndense_5_1/kernelPlaceholder_8*
use_locking( *
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:*
shape:
�
Assign_9Assigndense_5_1/biasPlaceholder_9*
use_locking( *
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:
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
SGD/iterations/readIdentitySGD/iterations*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
Y
SGD/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *
�#<
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
SGD/momentum/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
p
SGD/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/decay
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 
�
lambda_1_targetPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
r
lambda_1_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*'
_output_shapes
:���������*
T0
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
loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*#
_output_shapes
:���������*
T0
b
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
SGD_1/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
t
SGD_1/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
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
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
a
SGD_1/lr/readIdentitySGD_1/lr*
T0*
_class
loc:@SGD_1/lr*
_output_shapes
: 
a
SGD_1/momentum/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
r
SGD_1/momentum
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
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
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD_1/decay*
validate_shape(
j
SGD_1/decay/readIdentitySGD_1/decay*
_class
loc:@SGD_1/decay*
_output_shapes
: *
T0
�
lambda_1_target_1Placeholder*0
_output_shapes
:������������������*%
shape:������������������*
dtype0
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
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*#
_output_shapes
:���������*
T0
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss_1/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
]

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
i
y_truePlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
g
maskPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
Y

loss_2/subSublambda_1/suby_true*'
_output_shapes
:���������*
T0
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
loss_2/Absloss_2/Less/y*
T0*'
_output_shapes
:���������
U
loss_2/SquareSquare
loss_2/sub*'
_output_shapes
:���������*
T0
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
loss_2/sub_1/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
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
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*'
_output_shapes
:���������*
T0
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*'
_output_shapes
:���������*
T0
Z
loss_2/mul_2Mulloss_2/Selectmask*
T0*'
_output_shapes
:���������
g
loss_2/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
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
lambda_1_target_2Placeholder*0
_output_shapes
:������������������*%
shape:������������������*
dtype0
n
loss_sample_weightsPlaceholder*#
_output_shapes
:���������*
shape:���������*
dtype0
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
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:���������
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

loss_3/mulMulloss_3/mul/xloss_3/loss_loss/Mean_2*
_output_shapes
: *
T0
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
loss_3/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*#
_output_shapes
:���������*
T0
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

d
loss_3/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*#
_output_shapes
:���������*
T0
f
loss_3/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
S
loss_3/mul_1/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
a
loss_3/mul_1Mulloss_3/mul_1/xloss_3/lambda_1_loss/Mean_2*
_output_shapes
: *
T0
L

loss_3/addAdd
loss_3/mulloss_3/mul_1*
_output_shapes
: *
T0
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
4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
m
#metrics_2/mean_absolute_error/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
metrics_2/mean_q/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
�
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
IsVariableInitialized_25IsVariableInitializedSGD/iterations*
_output_shapes
: *!
_class
loc:@SGD/iterations*
dtype0	
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
IsVariableInitialized_29IsVariableInitializedSGD_1/iterations*
_output_shapes
: *#
_class
loc:@SGD_1/iterations*
dtype0	
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
IsVariableInitialized_32IsVariableInitializedSGD_1/decay*
_output_shapes
: *
_class
loc:@SGD_1/decay*
dtype0
�
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"��uF      ��w	l%��̩�AJ��
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
shape:���������*
dtype0*+
_output_shapes
:���������
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
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
Y
flatten_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
dtype0*
_output_shapes
:	�*
seed2���*
seed���)*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�*
	container *
shape:	�
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
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes
:	�*
T0*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
_output_shapes	
:�*
valueB�*    *
dtype0
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
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:�
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
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:����������
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
dense_2/random_uniform/minConst*
_output_shapes
: *
valueB
 *?�ʽ*
dtype0
_
dense_2/random_uniform/maxConst*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
_output_shapes
:	�d*
seed2���*
seed���)*
T0*
dtype0
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
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
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(
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
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������d*
T0
\
activation_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:���������d
m
dense_3/random_uniform/shapeConst*
_output_shapes
:*
valueB"d   2   *
dtype0
_
dense_3/random_uniform/minConst*
valueB
 *��L�*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2�ځ*
seed���)
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0
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
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0
{
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes

:d2
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
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:2*
use_locking(
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:2
�
dense_3/MatMulMatMulactivation_2/Reludense_3/kernel/read*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( *
T0
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
dense_4/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�D�>
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
VariableV2*
dtype0*
_output_shapes

:2*
	container *
shape
:2*
shared_name 
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
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_4/bias/readIdentitydense_4/bias*
_output_shapes
:*
T0*
_class
loc:@dense_4/bias
�
dense_4/MatMulMatMulactivation_3/Reludense_4/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
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
seed���)*
T0*
dtype0*
_output_shapes

:*
seed2��
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
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_5/kernel
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
dense_5/bias/readIdentitydense_5/bias*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:
�
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
m
lambda_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
o
lambda_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
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
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
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
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*'
_output_shapes
:���������*
T0
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
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
_
lambda_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:
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
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *o�9
k
Adam/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
Adam/beta_1
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1
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
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
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
flatten_1_1/ShapeShapeflatten_1_input_1*
_output_shapes
:*
T0*
out_type0
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
!flatten_1_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
�
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
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
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
_output_shapes
:*
T0*

axis *
N
�
flatten_1_1/ReshapeReshapeflatten_1_input_1flatten_1_1/stack*0
_output_shapes
:������������������*
T0*
Tshape0
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
dense_1_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�~�=
�
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�*
seed2��h
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
dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
T0*
_output_shapes
:	�
�
dense_1_1/kernel
VariableV2*
shape:	�*
shared_name *
dtype0*
_output_shapes
:	�*
	container 
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
dense_1_1/kernel/readIdentitydense_1_1/kernel*
_output_shapes
:	�*
T0*#
_class
loc:@dense_1_1/kernel
^
dense_1_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
|
dense_1_1/bias
VariableV2*
dtype0*
_output_shapes	
:�*
	container *
shape:�*
shared_name 
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
dense_1_1/bias/readIdentitydense_1_1/bias*
T0*!
_class
loc:@dense_1_1/bias*
_output_shapes	
:�
�
dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b( 
�
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
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
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
_output_shapes
:	�d*
T0
�
dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
T0*
_output_shapes
:	�d
�
dense_2_1/kernel
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
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
dense_2_1/kernel/readIdentitydense_2_1/kernel*
_output_shapes
:	�d*
T0*#
_class
loc:@dense_2_1/kernel
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
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
w
dense_2_1/bias/readIdentitydense_2_1/bias*
T0*!
_class
loc:@dense_2_1/bias*
_output_shapes
:d
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
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
dtype0*
_output_shapes

:d2*
seed2���*
seed���)*
T0
�
dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
_output_shapes

:d2*
T0
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
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0*#
_class
loc:@dense_3_1/kernel
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
VariableV2*
dtype0*
_output_shapes
:2*
	container *
shape:2*
shared_name 
�
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias
w
dense_3_1/bias/readIdentitydense_3_1/bias*
T0*!
_class
loc:@dense_3_1/bias*
_output_shapes
:2
�
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*
T0*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( 
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
dense_4_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"2      *
dtype0
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
&dense_4_1/random_uniform/RandomUniformRandomUniformdense_4_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:2*
seed2ֹM*
seed���)
�
dense_4_1/random_uniform/subSubdense_4_1/random_uniform/maxdense_4_1/random_uniform/min*
_output_shapes
: *
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes

:2*
	container *
shape
:2
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
dense_4_1/kernel/readIdentitydense_4_1/kernel*
T0*#
_class
loc:@dense_4_1/kernel*
_output_shapes

:2
\
dense_4_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_4_1/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
dense_4_1/bias/readIdentitydense_4_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_4_1/bias
�
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
dense_5_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
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
seed���)*
T0*
dtype0*
_output_shapes

:*
seed2��
�
dense_5_1/random_uniform/subSubdense_5_1/random_uniform/maxdense_5_1/random_uniform/min*
_output_shapes
: *
T0
�
dense_5_1/random_uniform/mulMul&dense_5_1/random_uniform/RandomUniformdense_5_1/random_uniform/sub*
_output_shapes

:*
T0
�
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
T0*
_output_shapes

:
�
dense_5_1/kernel
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
�
dense_5_1/bias/AssignAssigndense_5_1/biasdense_5_1/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@dense_5_1/bias
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
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:���������*
Index0*
T0
d
lambda_1_1/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*
T0*'
_output_shapes
:���������*

Tdim0
q
 lambda_1_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
s
"lambda_1_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������
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
valueB"       *
dtype0*
_output_shapes
:
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
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_3/bias
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_4/bias
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
IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
_output_shapes
: *
_class
loc:@Adam/beta_2*
dtype0
�
IsVariableInitialized_14IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_15IsVariableInitializeddense_1_1/kernel*
_output_shapes
: *#
_class
loc:@dense_1_1/kernel*
dtype0
�
IsVariableInitialized_16IsVariableInitializeddense_1_1/bias*!
_class
loc:@dense_1_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_17IsVariableInitializeddense_2_1/kernel*#
_class
loc:@dense_2_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_18IsVariableInitializeddense_2_1/bias*
_output_shapes
: *!
_class
loc:@dense_2_1/bias*
dtype0
�
IsVariableInitialized_19IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_20IsVariableInitializeddense_3_1/bias*
_output_shapes
: *!
_class
loc:@dense_3_1/bias*
dtype0
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
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*#
_class
loc:@dense_5_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*
dtype0*
_output_shapes
: *!
_class
loc:@dense_5_1/bias
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
_output_shapes
:	�*
shape:	�*
dtype0
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
Assign_1Assigndense_1_1/biasPlaceholder_1*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:�*
use_locking( 
`
Placeholder_2Placeholder*
shape:	�d*
dtype0*
_output_shapes
:	�d
�
Assign_2Assigndense_2_1/kernelPlaceholder_2*
use_locking( *
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d
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
Placeholder_4Placeholder*
shape
:d2*
dtype0*
_output_shapes

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
Placeholder_5Placeholder*
_output_shapes
:2*
shape:2*
dtype0
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
Assign_6Assigndense_4_1/kernelPlaceholder_6*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2*
use_locking( *
T0
V
Placeholder_7Placeholder*
_output_shapes
:*
shape:*
dtype0
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
Placeholder_8Placeholder*
shape
:*
dtype0*
_output_shapes

:
�
Assign_8Assigndense_5_1/kernelPlaceholder_8*
use_locking( *
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:*
shape:
�
Assign_9Assigndense_5_1/biasPlaceholder_9*
use_locking( *
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
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
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
use_locking(*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: 
s
SGD/iterations/readIdentitySGD/iterations*
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
Y
SGD/lr/initial_valueConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0
j
SGD/lr
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: *
use_locking(
m
SGD/momentum/readIdentitySGD/momentum*
_class
loc:@SGD/momentum*
_output_shapes
: *
T0
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/decay
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
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB 
�
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*#
_output_shapes
:���������*
T0
b
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
W
loss/mulMul
loss/mul/xloss/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
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
SGD_1/iterations/readIdentitySGD_1/iterations*#
_class
loc:@SGD_1/iterations*
_output_shapes
: *
T0	
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
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@SGD_1/momentum
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
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD_1/decay*
validate_shape(*
_output_shapes
: 
j
SGD_1/decay/readIdentitySGD_1/decay*
T0*
_class
loc:@SGD_1/decay*
_output_shapes
: 
�
lambda_1_target_1Placeholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
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
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
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
loss_1/lambda_1_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*#
_output_shapes
:���������*
T0
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
d
loss_1/lambda_1_loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss_1/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
]

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
T0*
_output_shapes
: 
i
y_truePlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
g
maskPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
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
loss_2/sub*'
_output_shapes
:���������*
T0
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
loss_2/sub*'
_output_shapes
:���������*
T0
S
loss_2/sub_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
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
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*'
_output_shapes
:���������*
T0
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*'
_output_shapes
:���������*
T0
Z
loss_2/mul_2Mulloss_2/Selectmask*'
_output_shapes
:���������*
T0
g
loss_2/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
�

loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
�
loss_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
�
lambda_1_target_2Placeholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
n
loss_sample_weightsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
t
lambda_1_sample_weights_2Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB 
�
loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*#
_output_shapes
:���������*
T0
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
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:���������
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Q
loss_3/mul/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
loss_3/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
loss_3/mulloss_3/mul_1*
_output_shapes
: *
T0
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
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
IsVariableInitialized_27IsVariableInitializedSGD/momentum*
_output_shapes
: *
_class
loc:@SGD/momentum*
dtype0

IsVariableInitialized_28IsVariableInitialized	SGD/decay*
_output_shapes
: *
_class
loc:@SGD/decay*
dtype0
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
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign""�
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
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0"�
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
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0��j%.       ��W�	X��̩�A*#
!
Average reward per episode�������,       ���E	���̩�A*!

total reward per episode  �	=M-       <A��	G�̩�A* 

Average reward per step�����T       `/�#	�̩�A*

epsilon�����5�-       <A��	���̩�A* 

Average reward per step����ld�       `/�#	y��̩�A*

epsilon���?�m�-       <A��	{��̩�A* 

Average reward per step������T       `/�#		��̩�A*

epsilon������-       <A��	���̩�A * 

Average reward per step������       `/�#	���̩�A *

epsilon���jcP-       <A��	=��̩�A!* 

Average reward per step������+       `/�#	��̩�A!*

epsilon���6g	-       <A��	��̩�A"* 

Average reward per step������       `/�#	U�̩�A"*

epsilon���u�!-       <A��	���̩�A#* 

Average reward per step���唇       `/�#	���̩�A#*

epsilon����ƙ-       <A��	�+�̩�A$* 

Average reward per step���)=y       `/�#	o,�̩�A$*

epsilon�����P�-       <A��	R&�̩�A%* 

Average reward per step���~��t       `/�#	�&�̩�A%*

epsilon�����;�-       <A��	B
�̩�A&* 

Average reward per step����iEu       `/�#	g�̩�A&*

epsilon����]z�-       <A��	*��̩�A'* 

Average reward per step������       `/�#	��̩�A'*

epsilon����;�-       <A��	H!�̩�A(* 

Average reward per step����DE�       `/�#	�!�̩�A(*

epsilon������-       <A��	��"�̩�A)* 

Average reward per step���u�        `/�#	X�"�̩�A)*

epsilon����ɦ
-       <A��	m:$�̩�A** 

Average reward per step���mn*�       `/�#	;$�̩�A**

epsilon���Q+��-       <A��	�"&�̩�A+* 

Average reward per step����&&       `/�#	�#&�̩�A+*

epsilon����-3�-       <A��	[]'�̩�A,* 

Average reward per step����;
v       `/�#	 ^'�̩�A,*

epsilon������R-       <A��	~9)�̩�A-* 

Average reward per step���"z�       `/�#	:)�̩�A-*

epsilon�����r-       <A��	+�̩�A.* 

Average reward per step����	l       `/�#	�+�̩�A.*

epsilon������-       <A��	L,�̩�A/* 

Average reward per step���:,�       `/�#	�L,�̩�A/*

epsilon����e*-       <A��	m .�̩�A0* 

Average reward per step�����?       `/�#	� .�̩�A0*

epsilon���bL(�-       <A��	�M/�̩�A1* 

Average reward per step�������       `/�#	8N/�̩�A1*

epsilon���D�D�-       <A��	*1�̩�A2* 

Average reward per step���]���       `/�#	�*1�̩�A2*

epsilon�����&�-       <A��	�a2�̩�A3* 

Average reward per step���A/��       `/�#	�b2�̩�A3*

epsilon������-       <A��	�C4�̩�A4* 

Average reward per step����Tz�       `/�#	[D4�̩�A4*

epsilon���(���-       <A��	�36�̩�A5* 

Average reward per step����(MQ       `/�#	a46�̩�A5*

epsilon����^H�-       <A��	<l7�̩�A6* 

Average reward per step���}i�       `/�#		m7�̩�A6*

epsilon���Ӡ\Q-       <A��	F9�̩�A7* 

Average reward per step���<\j       `/�#	�F9�̩�A7*

epsilon���m�rT-       <A��	qv:�̩�A8* 

Average reward per step���V��       `/�#	w:�̩�A8*

epsilon���+�� -       <A��	mY<�̩�A9* 

Average reward per step���
=4;       `/�#	Z<�̩�A9*

epsilon����6�-       <A��	�6>�̩�A:* 

Average reward per step���8a�       `/�#	a7>�̩�A:*

epsilon���_��?-       <A��	p?�̩�A;* 

Average reward per step����[�       `/�#	�p?�̩�A;*

epsilon���?�Nh-       <A��	TUA�̩�A<* 

Average reward per step���#��       `/�#	!VA�̩�A<*

epsilon��� -�G-       <A��	y�B�̩�A=* 

Average reward per step���@�o[       `/�#	�B�̩�A=*

epsilon���#/�u0       ���_	L�B�̩�A*#
!
Average reward per episode>x����.       ��W�	ũB�̩�A*!

total reward per episode   ����-       <A��	��F�̩�A>* 

Average reward per step>x��A7c       `/�#	��F�̩�A>*

epsilon>x���9�-       <A��	�uH�̩�A?* 

Average reward per step>x��4��       `/�#	KvH�̩�A?*

epsilon>x��A�V-       <A��	�I�̩�A@* 

Average reward per step>x�)r
<       `/�#	ѮI�̩�A@*

epsilon>x�N��x-       <A��	?�K�̩�AA* 

Average reward per step>x���       `/�#	X�K�̩�AA*

epsilon>x����-       <A��	&rM�̩�AB* 

Average reward per step>x��hY[       `/�#	�rM�̩�AB*

epsilon>x�u�t1-       <A��	W�N�̩�AC* 

Average reward per step>x����       `/�#	�N�̩�AC*

epsilon>x�8G.-       <A��	S�P�̩�AD* 

Average reward per step>x����       `/�#	�P�̩�AD*

epsilon>x�����-       <A��	G�Q�̩�AE* 

Average reward per step>x��/�       `/�#	�Q�̩�AE*

epsilon>x�̽�-       <A��	 �S�̩�AF* 

Average reward per step>x��ӑ�       `/�#	�S�̩�AF*

epsilon>x����P-       <A��	��U�̩�AG* 

Average reward per step>x�b8:j       `/�#	B�U�̩�AG*

epsilon>x�9���-       <A��	��V�̩�AH* 

Average reward per step>x���"t       `/�#	Q�V�̩�AH*

epsilon>x�Q'@�-       <A��	��X�̩�AI* 

Average reward per step>x�� (|       `/�#	��X�̩�AI*

epsilon>x�=?�-       <A��	��Z�̩�AJ* 

Average reward per step>x�v�t#       `/�#	5�Z�̩�AJ*

epsilon>x����8-       <A��	t�\�̩�AK* 

Average reward per step>x���k       `/�#	J�\�̩�AK*

epsilon>x��>B�-       <A��	9�]�̩�AL* 

Average reward per step>x�JC|�       `/�#	�]�̩�AL*

epsilon>x��-       <A��	u�_�̩�AM* 

Average reward per step>x�����       `/�#	�_�̩�AM*

epsilon>x�e�3�-       <A��	ӡa�̩�AN* 

Average reward per step>x�(�)�       `/�#	��a�̩�AN*

epsilon>x�pd-       <A��	��b�̩�AO* 

Average reward per step>x��57)       `/�#	4�b�̩�AO*

epsilon>x��m��-       <A��	�d�̩�AP* 

Average reward per step>x��t��       `/�#	��d�̩�AP*

epsilon>x�n܂�-       <A��	-�e�̩�AQ* 

Average reward per step>x��Z�%       `/�#	��e�̩�AQ*

epsilon>x��cCD-       <A��	��g�̩�AR* 

Average reward per step>x��)�       `/�#	@�g�̩�AR*

epsilon>x�&�%R-       <A��	��i�̩�AS* 

Average reward per step>x��a2       `/�#	|�i�̩�AS*

epsilon>x��X �-       <A��	��k�̩�AT* 

Average reward per step>x�Soe�       `/�#	��k�̩�AT*

epsilon>x� ��|-       <A��	�m�̩�AU* 

Average reward per step>x���       `/�#	�m�̩�AU*

epsilon>x���-       <A��	,�n�̩�AV* 

Average reward per step>x��F.�       `/�#	��n�̩�AV*

epsilon>x��%�R-       <A��	��p�̩�AW* 

Average reward per step>x��!��       `/�#	��p�̩�AW*

epsilon>x���H-       <A��	� r�̩�AX* 

Average reward per step>x�լ!Y       `/�#	�r�̩�AX*

epsilon>x���	�-       <A��	G�s�̩�AY* 

Average reward per step>x��&k�       `/�#	�s�̩�AY*

epsilon>x��U-       <A��	~u�̩�AZ* 

Average reward per step>x��C��       `/�#	Ku�̩�AZ*

epsilon>x�T��9-       <A��	mw�̩�A[* 

Average reward per step>x��'v2       `/�#	w�̩�A[*

epsilon>x��Qk-       <A��	e�x�̩�A\* 

Average reward per step>x��E&�       `/�#	��x�̩�A\*

epsilon>x��� �-       <A��	z�̩�A]* 

Average reward per step>x����       `/�#	�z�̩�A]*

epsilon>x��Z-       <A��	T�{�̩�A^* 

Average reward per step>x�>\#       `/�#	" |�̩�A^*

epsilon>x���W0       ���_		|�̩�A*#
!
Average reward per episode������+�.       ��W�	�|�̩�A*!

total reward per episode  Ì	u�-       <A��	�?�̩�A_* 

Average reward per step����3�O�       `/�#	d@�̩�A_*

epsilon����M�c�-       <A��	�#��̩�A`* 

Average reward per step����YhB       `/�#	d$��̩�A`*

epsilon����pU��-       <A��	���̩�Aa* 

Average reward per step������<       `/�#	���̩�Aa*

epsilon�������-       <A��	?��̩�Ab* 

Average reward per step������9       `/�#	�?��̩�Ab*

epsilon����w�-       <A��	�+��̩�Ac* 

Average reward per step����#6!
       `/�#	Z,��̩�Ac*

epsilon���� 	[�-       <A��	���̩�Ad* 

Average reward per step����ig��       `/�#	y��̩�Ad*

epsilon����X0-       <A��	�U��̩�Ae* 

Average reward per step����X�W�       `/�#	�V��̩�Ae*

epsilon����e��-       <A��	�/��̩�Af* 

Average reward per step������:�       `/�#	w0��̩�Af*

epsilon����C��I-       <A��	m��̩�Ag* 

Average reward per step����[a~       `/�#	�m��̩�Ag*

epsilon����ULS�-       <A��	�[��̩�Ah* 

Average reward per step����b�S       `/�#	t\��̩�Ah*

epsilon����ݶ�{-       <A��	kF��̩�Ai* 

Average reward per step����� �W       `/�#	G��̩�Ai*

epsilon����#�-       <A��	]5��̩�Aj* 

Average reward per step������43       `/�#	36��̩�Aj*

epsilon������F,-       <A��	�z��̩�Ak* 

Average reward per step����C�3K       `/�#	W{��̩�Ak*

epsilon�����"�v-       <A��	�_��̩�Al* 

Average reward per step�������       `/�#	�`��̩�Al*

epsilon�����5S�-       <A��	@M��̩�Am* 

Average reward per step�������>       `/�#	�M��̩�Am*

epsilon�����*p�-       <A��	����̩�An* 

Average reward per step����Qb       `/�#	:���̩�An*

epsilon�����
-       <A��	�i��̩�Ao* 

Average reward per step����Gh       `/�#	�j��̩�Ao*

epsilon����1��-       <A��	�L��̩�Ap* 

Average reward per step����=g�       `/�#	�M��̩�Ap*

epsilon������ݕ-       <A��	M���̩�Aq* 

Average reward per step����vŉ�       `/�#	���̩�Aq*

epsilon������HO-       <A��	����̩�Ar* 

Average reward per step������K       `/�#	����̩�Ar*

epsilon����ꔶn-       <A��	����̩�As* 

Average reward per step�����qh0       `/�#	����̩�As*

epsilon����
>�_-       <A��	]o��̩�At* 

Average reward per step�������       `/�#	�o��̩�At*

epsilon����� �-       <A��	e���̩�Au* 

Average reward per step����O]�6       `/�#	;���̩�Au*

epsilon������Z�-       <A��	܂��̩�Av* 

Average reward per step����[
�       `/�#	����̩�Av*

epsilon����TH9-       <A��	Lk��̩�Aw* 

Average reward per step������?       `/�#	l��̩�Aw*

epsilon������>�-       <A��	ȳ��̩�Ax* 

Average reward per step������&�       `/�#	����̩�Ax*

epsilon����-�z-       <A��	����̩�Ay* 

Average reward per step����<H�&       `/�#	Z���̩�Ay*

epsilon����4���-       <A��	�ͭ�̩�Az* 

Average reward per step����Q5�       `/�#	Fέ�̩�Az*

epsilon������ %-       <A��	%���̩�A{* 

Average reward per step�����	XS       `/�#	����̩�A{*

epsilon����*��C-       <A��	u���̩�A|* 

Average reward per step������j       `/�#	���̩�A|*

epsilon������-       <A��	=Ѳ�̩�A}* 

Average reward per step����Ռ�       `/�#	�Ѳ�̩�A}*

epsilon�������-       <A��	����̩�A~* 

Average reward per step����e���       `/�#	T���̩�A~*

epsilon����7�7�-       <A��	ᵅ̩�A* 

Average reward per step����g�̆       `/�#	�ᵅ̩�A*

epsilon�����2u.       ��W�	�ɷ�̩�A�* 

Average reward per step����齑�       ��2	�ʷ�̩�A�*

epsilon����d�!%.       ��W�	5��̩�A�* 

Average reward per step����6%�       ��2	���̩�A�*

epsilon����nA �.       ��W�	��̩�A�* 

Average reward per step����z2?I       ��2	���̩�A�*

epsilon�����.       ��W�	���̩�A�* 

Average reward per step�����J�        ��2	l��̩�A�*

epsilon�����2��.       ��W�	C澅̩�A�* 

Average reward per step��������       ��2	�澅̩�A�*

epsilon�����c�|.       ��W�	�$��̩�A�* 

Average reward per step����4���       ��2	�%��̩�A�*

epsilon����	G0�.       ��W�	)$̩�A�* 

Average reward per step�����Ϛ       ��2	%̩�A�*

epsilon����Y}�_.       ��W�	i8ą̩�A�* 

Average reward per step������h       ��2	29ą̩�A�*

epsilon����[�s.       ��W�	�&ƅ̩�A�* 

Average reward per step����{!p       ��2	�'ƅ̩�A�*

epsilon����OT��.       ��W�	�'ȅ̩�A�* 

Average reward per step������e�       ��2	�(ȅ̩�A�*

epsilon����рȽ.       ��W�	w/ʅ̩�A�* 

Average reward per step�������t       ��2	+0ʅ̩�A�*

epsilon�����rq�.       ��W�	�:̩̅�A�* 

Average reward per step����-�7�       ��2	�;̩̅�A�*

epsilon����E��E.       ��W�	�=΅̩�A�* 

Average reward per step�����A�       ��2	�>΅̩�A�*

epsilon����Z��.       ��W�	z�υ̩�A�* 

Average reward per step�����`C       ��2	G�υ̩�A�*

epsilon������.       ��W�	��х̩�A�* 

Average reward per step����{ʅ       ��2	V�х̩�A�*

epsilon�����w�+.       ��W�	;�Ӆ̩�A�* 

Average reward per step�����G       ��2	ʉӅ̩�A�*

epsilon����|���.       ��W�	rlՅ̩�A�* 

Average reward per step������pT       ��2	mՅ̩�A�*

epsilon�����8
.       ��W�	'lׅ̩�A�* 

Average reward per step����'C�       ��2	�lׅ̩�A�*

epsilon������͠.       ��W�	<Pم̩�A�* 

Average reward per step�����P�N       ��2		Qم̩�A�*

epsilon����L���.       ��W�	yۅ̩�A�* 

Average reward per step����IG       ��2	zۅ̩�A�*

epsilon������v.       ��W�	�'ޅ̩�A�* 

Average reward per step����Tw�       ��2	V(ޅ̩�A�*

epsilon����ǅ6�.       ��W�	m߅̩�A�* 

Average reward per step�������       ��2		n߅̩�A�*

epsilon������I.       ��W�	��̩�A�* 

Average reward per step����3F��       ��2	��̩�A�*

epsilon����h4�o.       ��W�	�p�̩�A�* 

Average reward per step���� �       ��2	~q�̩�A�*

epsilon����Ax#{.       ��W�	��̩�A�* 

Average reward per step����kJ��       ��2	���̩�A�*

epsilon�������.       ��W�	 ��̩�A�* 

Average reward per step������       ��2	���̩�A�*

epsilon������Ւ.       ��W�	U��̩�A�* 

Average reward per step�����E�       ��2	��̩�A�*

epsilon�������y.       ��W�	P��̩�A�* 

Average reward per step������wl       ��2	��̩�A�*

epsilon����^b7.       ��W�	z��̩�A�* 

Average reward per step�����#��       ��2	&��̩�A�*

epsilon�����Y��.       ��W�	*��̩�A�* 

Average reward per step����Y��j       ��2	���̩�A�*

epsilon�����mS.       ��W�	B���̩�A�* 

Average reward per step��������       ��2	���̩�A�*

epsilon����A�c}.       ��W�	���̩�A�* 

Average reward per step����a�p�       ��2	g��̩�A�*

epsilon����b��z.       ��W�	���̩�A�* 

Average reward per step������s�       ��2	c��̩�A�*

epsilon�����HuQ.       ��W�	��̩�A�* 

Average reward per step�����I.f       ��2	���̩�A�*

epsilon������EW.       ��W�	U/��̩�A�* 

Average reward per step����!G�       ��2	0��̩�A�*

epsilon����Z>@.       ��W�	�[��̩�A�* 

Average reward per step����.�M�       ��2	}\��̩�A�*

epsilon����fS��.       ��W�	�X��̩�A�* 

Average reward per step�������D       ��2	`Y��̩�A�*

epsilon����� ��.       ��W�	A��̩�A�* 

Average reward per step����֎
�       ��2	�A��̩�A�*

epsilon�����;B.       ��W�	3P�̩�A�* 

Average reward per step����[�=       ��2	Q�̩�A�*

epsilon�������j.       ��W�	���̩�A�* 

Average reward per step����X���       ��2	X��̩�A�*

epsilon����m(��0       ���_	��̩�A*#
!
Average reward per episode*T(�G3&.       ��W�	���̩�A*!

total reward per episode  @ONs.       ��W�	M��̩�A�* 

Average reward per step*T(�>8e�       ��2	��̩�A�*

epsilon*T(��6[�.       ��W�	zq�̩�A�* 

Average reward per step*T(�3,.�       ��2	Gr�̩�A�*

epsilon*T(�u|�*.       ��W�	![
�̩�A�* 

Average reward per step*T(�Q� �       ��2	�[
�̩�A�*

epsilon*T(���.       ��W�	���̩�A�* 

Average reward per step*T(�{�6       ��2	J��̩�A�*

epsilon*T(���Z.       ��W�	�q�̩�A�* 

Average reward per step*T(����       ��2	Kr�̩�A�*

epsilon*T(�-�m~.       ��W�	�d�̩�A�* 

Average reward per step*T(�X���       ��2	se�̩�A�*

epsilon*T(��*�.       ��W�	&��̩�A�* 

Average reward per step*T(�?=8�       ��2	ۧ�̩�A�*

epsilon*T(�#Y�.       ��W�	���̩�A�* 

Average reward per step*T(��
�	       ��2	P��̩�A�*

epsilon*T(�O)��.       ��W�	���̩�A�* 

Average reward per step*T(�ey�)       ��2	3��̩�A�*

epsilon*T(����.       ��W�	ƨ�̩�A�* 

Average reward per step*T(�����       ��2	v��̩�A�*

epsilon*T(�����.       ��W�	˄�̩�A�* 

Average reward per step*T(�>�!       ��2	Y��̩�A�*

epsilon*T(��n��.       ��W�	{��̩�A�* 

Average reward per step*T(�~�       ��2	@��̩�A�*

epsilon*T(���X�.       ��W�	���̩�A�* 

Average reward per step*T(�^�.       ��2	��̩�A�*

epsilon*T(��r}.       ��W�	<��̩�A�* 

Average reward per step*T(�ۖ��       ��2	���̩�A�*

epsilon*T(�����.       ��W�	|��̩�A�* 

Average reward per step*T(�w��3       ��2	��̩�A�*

epsilon*T(����.       ��W�	\��̩�A�* 

Average reward per step*T(�AF�       ��2	��̩�A�*

epsilon*T(���M.       ��W�	� �̩�A�* 

Average reward per step*T(�D       ��2	�� �̩�A�*

epsilon*T(�!/,�.       ��W�	��"�̩�A�* 

Average reward per step*T(�Yd�m       ��2	f�"�̩�A�*

epsilon*T(�`�.       ��W�	��$�̩�A�* 

Average reward per step*T(����8       ��2	��$�̩�A�*

epsilon*T(�z�Ph.       ��W�	��%�̩�A�* 

Average reward per step*T(�.2�U       ��2	C�%�̩�A�*

epsilon*T(��h�.       ��W�	U�'�̩�A�* 

Average reward per step*T(�-<6`       ��2	��'�̩�A�*

epsilon*T(��^G�.       ��W�	��)�̩�A�* 

Average reward per step*T(��2�^       ��2	^�)�̩�A�*

epsilon*T(�����.       ��W�	~�*�̩�A�* 

Average reward per step*T(�� ��       ��2	" +�̩�A�*

epsilon*T(���/.       ��W�	��,�̩�A�* 

Average reward per step*T(��m�       ��2	j�,�̩�A�*

epsilon*T(���O�.       ��W�	�".�̩�A�* 

Average reward per step*T(�5l�(       ��2	�#.�̩�A�*

epsilon*T(����.       ��W�	s0�̩�A�* 

Average reward per step*T(�`�       ��2	
0�̩�A�*

epsilon*T(��J/.       ��W�	2;2�̩�A�* 

Average reward per step*T(���)p       ��2	><2�̩�A�*

epsilon*T(����.       ��W�	�+4�̩�A�* 

Average reward per step*T(�3Y�*       ��2	|,4�̩�A�*

epsilon*T(��5{�0       ���_	�H4�̩�A*#
!
Average reward per episode  �����.       ��W�	I4�̩�A*!

total reward per episode  �b{.       ��W�	�8�̩�A�* 

Average reward per step  ��ws0Q       ��2	�8�̩�A�*

epsilon  ��i%�.       ��W�	�:�̩�A�* 

Average reward per step  ����B]       ��2	�:�̩�A�*

epsilon  ��ў��.       ��W�	@�;�̩�A�* 

Average reward per step  ������       ��2	��;�̩�A�*

epsilon  �����.       ��W�	74=�̩�A�* 

Average reward per step  ��HGG�       ��2	�4=�̩�A�*

epsilon  ����NJ.       ��W�	V?�̩�A�* 

Average reward per step  ��8�]/       ��2	?�̩�A�*

epsilon  ��?�I.       ��W�	�U@�̩�A�* 

Average reward per step  ����y�       ��2	�V@�̩�A�*

epsilon  ����6.       ��W�	p^B�̩�A�* 

Average reward per step  ����       ��2	)_B�̩�A�*

epsilon  ��FJ]�.       ��W�	gD�̩�A�* 

Average reward per step  ��oe?�       ��2	8hD�̩�A�*

epsilon  ���׫.       ��W�	�nF�̩�A�* 

Average reward per step  ���Jq�       ��2	�oF�̩�A�*

epsilon  ����.       ��W�	�H�̩�A�* 

Average reward per step  ��!���       ��2	��H�̩�A�*

epsilon  ����
^.       ��W�	�J�̩�A�* 

Average reward per step  ��w1�s       ��2	��J�̩�A�*

epsilon  �����T.       ��W�	ĔL�̩�A�* 

Average reward per step  ��R�       ��2	_�L�̩�A�*

epsilon  ����I.       ��W�	ճN�̩�A�* 

Average reward per step  ��tFr0       ��2	l�N�̩�A�*

epsilon  ��ߩ�8.       ��W�	��P�̩�A�* 

Average reward per step  �� 7X;       ��2	�P�̩�A�*

epsilon  ��^�9.       ��W�	��R�̩�A�* 

Average reward per step  ��jl�       ��2	@�R�̩�A�*

epsilon  ��販Q.       ��W�	�)T�̩�A�* 

Average reward per step  ��.�)3       ��2	V*T�̩�A�*

epsilon  ���K7.       ��W�	�U�̩�A�* 

Average reward per step  ���f�>       ��2	=�U�̩�A�*

epsilon  ����9.       ��W�	�qW�̩�A�* 

Average reward per step  ��rH��       ��2	*rW�̩�A�*

epsilon  ��C���.       ��W�	<�X�̩�A�* 

Average reward per step  ���ҿ5       ��2	8�X�̩�A�*

epsilon  ���ӟ�.       ��W�	��Z�̩�A�* 

Average reward per step  ���d�h       ��2	��Z�̩�A�*

epsilon  ��C�k.       ��W�	��\�̩�A�* 

Average reward per step  ��S+�z       ��2	d�\�̩�A�*

epsilon  ��Kj�.       ��W�	��^�̩�A�* 

Average reward per step  ��Ī��       ��2	�^�̩�A�*

epsilon  ��J��e.       ��W�	�`�̩�A�* 

Average reward per step  ��0�'#       ��2	��`�̩�A�*

epsilon  ��$���.       ��W�	+�b�̩�A�* 

Average reward per step  ��:>�b       ��2	Ƣb�̩�A�*

epsilon  ���W.       ��W�	`�c�̩�A�* 

Average reward per step  ��=�[L       ��2	��c�̩�A�*

epsilon  ����.       ��W�	R�e�̩�A�* 

Average reward per step  ���
�        ��2	�e�̩�A�*

epsilon  ��z��.       ��W�	e�g�̩�A�* 

Average reward per step  ��l͡       ��2	��g�̩�A�*

epsilon  ���1��.       ��W�	��i�̩�A�* 

Average reward per step  ���~��       ��2	��i�̩�A�*

epsilon  ����_�.       ��W�	��k�̩�A�* 

Average reward per step  �����       ��2	e�k�̩�A�*

epsilon  ���W^�.       ��W�	[�m�̩�A�* 

Average reward per step  ��O�~       ��2	��m�̩�A�*

epsilon  ��(0�.       ��W�	�o�̩�A�* 

Average reward per step  ��.U��       ��2	uo�̩�A�*

epsilon  ����
w.       ��W�	(�p�̩�A�* 

Average reward per step  ���k�       ��2	��p�̩�A�*

epsilon  ���$�.       ��W�	g�r�̩�A�* 

Average reward per step  �����       ��2	�r�̩�A�*

epsilon  ����<~.       ��W�	�t�̩�A�* 

Average reward per step  ��L��       ��2	?t�̩�A�*

epsilon  ��bPT�.       ��W�	��u�̩�A�* 

Average reward per step  ��/��       ��2	��u�̩�A�*

epsilon  �����.       ��W�	��w�̩�A�* 

Average reward per step  ����AL       ��2	~�w�̩�A�*

epsilon  �����W.       ��W�	'y�̩�A�* 

Average reward per step  ��9B�       ��2	�'y�̩�A�*

epsilon  ��
�M�.       ��W�	h[{�̩�A�* 

Average reward per step  ����@V       ��2	\{�̩�A�*

epsilon  ���6P.       ��W�	'�}�̩�A�* 

Average reward per step  ���Y]�       ��2		�}�̩�A�*

epsilon  ����ǂ.       ��W�	%>�̩�A�* 

Average reward per step  ��䷪�       ��2	�>�̩�A�*

epsilon  ���\�.       ��W�	@j��̩�A�* 

Average reward per step  �����       ��2	�j��̩�A�*

epsilon  ��*Y�.       ��W�	g���̩�A�* 

Average reward per step  ��GJ       ��2	M���̩�A�*

epsilon  ���D-�.       ��W�	׆��̩�A�* 

Average reward per step  ��u<*       ��2	f���̩�A�*

epsilon  ��L^8�.       ��W�	����̩�A�* 

Average reward per step  ���^1M       ��2	ɪ��̩�A�*

epsilon  ��W��.       ��W�	^���̩�A�* 

Average reward per step  ��f��       ��2	���̩�A�*

epsilon  �����.       ��W�	Ƈ��̩�A�* 

Average reward per step  ������       ��2	]���̩�A�*

epsilon  ���w��.       ��W�	�{��̩�A�* 

Average reward per step  ��!g�$       ��2	�|��̩�A�*

epsilon  ��3s[.       ��W�	|`��̩�A�* 

Average reward per step  ����       ��2	a��̩�A�*

epsilon  ��.(G+.       ��W�	_��̩�A�* 

Average reward per step  �����       ��2	9��̩�A�*

epsilon  ��}�!.       ��W�	�K��̩�A�* 

Average reward per step  ��@�       ��2	�L��̩�A�*

epsilon  ��xԐ�.       ��W�	G���̩�A�* 

Average reward per step  ��n��Z       ��2	�̩�A�*

epsilon  ���+u.       ��W�	|��̩�A�* 

Average reward per step  ���(��       ��2	�|��̩�A�*

epsilon  ��9��.       ��W�	B|��̩�A�* 

Average reward per step  ���X�       ��2	5}��̩�A�*

epsilon  ����/.       ��W�	ٚ�̩�A�* 

Average reward per step  ��Sl�       ��2	�ٚ�̩�A�*

epsilon  �����.       ��W�	>М�̩�A�* 

Average reward per step  ��j��u       ��2	ќ�̩�A�*

epsilon  ���1�h.       ��W�	!Ξ�̩�A�* 

Average reward per step  ��{9T       ��2	_Ϟ�̩�A�*

epsilon  ��E��+.       ��W�	�̩�A�* 

Average reward per step  ��g�'�       ��2	��̩�A�*

epsilon  ����$s.       ��W�	t��̩�A�* 

Average reward per step  ��4��       ��2	 ��̩�A�*

epsilon  ��0�B
.       ��W�	R��̩�A�* 

Average reward per step  ��l-�       ��2	��̩�A�*

epsilon  ��!!OH.       ��W�	����̩�A�* 

Average reward per step  ���)y?       ��2	����̩�A�*

epsilon  �����.       ��W�	�(��̩�A�* 

Average reward per step  ����w�       ��2	|)��̩�A�*

epsilon  ��:���.       ��W�	���̩�A�* 

Average reward per step  ��}�b`       ��2	��̩�A�*

epsilon  ���e|.       ��W�	d<��̩�A�* 

Average reward per step  ���#J�       ��2	>=��̩�A�*

epsilon  ���VhH.       ��W�	u:��̩�A�* 

Average reward per step  ��V�s�       ��2	;��̩�A�*

epsilon  ����e�.       ��W�	�b��̩�A�* 

Average reward per step  ��$��       ��2	�c��̩�A�*

epsilon  ���֩&.       ��W�	x��̩�A�* 

Average reward per step  ���"�       ��2	g���̩�A�*

epsilon  ��"}�9.       ��W�	Eִ�̩�A�* 

Average reward per step  ���8d       ��2	0״�̩�A�*

epsilon  ��؏��.       ��W�	h$��̩�A�* 

Average reward per step  ��2�9�       ��2	>%��̩�A�*

epsilon  ��P��.       ��W�	p$��̩�A�* 

Average reward per step  ���X�       ��2	F%��̩�A�*

epsilon  ����V�.       ��W�	L��̩�A�* 

Average reward per step  ������       ��2	���̩�A�*

epsilon  ��1�ŕ0       ���_	�4��̩�A*#
!
Average reward per episode��+� x�.       ��W�	n5��̩�A*!

total reward per episode  <�֝2=.       ��W�	�(��̩�A�* 

Average reward per step��+��ӊ�       ��2	R)��̩�A�*

epsilon��+�{5�.       ��W�	.���̩�A�* 

Average reward per step��+�ၰ�       ��2	����̩�A�*

epsilon��+���j�.       ��W�	���̩�A�* 

Average reward per step��+����       ��2	���̩�A�*

epsilon��+��O�.       ��W�	�Æ̩�A�* 

Average reward per step��+��*�       ��2	�Æ̩�A�*

epsilon��+�v$��.       ��W�	9�ņ̩�A�* 

Average reward per step��+�M	��       ��2	�ņ̩�A�*

epsilon��+�֝�.       ��W�	��ǆ̩�A�* 

Average reward per step��+���       ��2	��ǆ̩�A�*

epsilon��+��M~w.       ��W�	��Ɇ̩�A�* 

Average reward per step��+�kո�       ��2	��Ɇ̩�A�*

epsilon��+��wJ.       ��W�	C�ˆ̩�A�* 

Average reward per step��+�g�<       ��2	�ˆ̩�A�*

epsilon��+���.       ��W�	��̩͆�A�* 

Average reward per step��+��7       ��2	4�̩͆�A�*

epsilon��+��D�.       ��W�	��φ̩�A�* 

Average reward per step��+����       ��2	��φ̩�A�*

epsilon��+�mYc�.       ��W�	�ц̩�A�* 

Average reward per step��+��ļ�       ��2	¥ц̩�A�*

epsilon��+��m0.       ��W�	��ӆ̩�A�* 

Average reward per step��+�
:�       ��2	r�ӆ̩�A�*

epsilon��+�E6.       ��W�	�Ն̩�A�* 

Average reward per step��+��`A�       ��2	�Ն̩�A�*

epsilon��+��?.       ��W�	*׆̩�A�* 

Average reward per step��+�.�       ��2	׆̩�A�*

epsilon��+�����.       ��W�	-"ن̩�A�* 

Average reward per step��+�*j�       ��2	�"ن̩�A�*

epsilon��+��!N.       ��W�	6"ۆ̩�A�* 

Average reward per step��+�z��       ��2	�"ۆ̩�A�*

epsilon��+��Y*.       ��W�	�+݆̩�A�* 

Average reward per step��+����       ��2	�,݆̩�A�*

epsilon��+��\&.       ��W�	�:߆̩�A�* 

Average reward per step��+���5�       ��2	�;߆̩�A�*

epsilon��+�g��.       ��W�	�3�̩�A�* 

Average reward per step��+���3       ��2	4�̩�A�*

epsilon��+���b.       ��W�	Q1�̩�A�* 

Average reward per step��+�ρ��       ��2	#2�̩�A�*

epsilon��+�cL�:.       ��W�	�#�̩�A�* 

Average reward per step��+�[Y�_       ��2	�$�̩�A�*

epsilon��+�#S�.       ��W�	-�̩�A�* 

Average reward per step��+��9�       ��2	�-�̩�A�*

epsilon��+�u.�%.       ��W�	k+�̩�A�* 

Average reward per step��+�Z��       ��2	A,�̩�A�*

epsilon��+��bp.       ��W�		��̩�A�* 

Average reward per step��+�����       ��2	���̩�A�*

epsilon��+�,�#�.       ��W�	;�̩�A�* 

Average reward per step��+��k�&       ��2	�;�̩�A�*

epsilon��+�(��.       ��W�	v��̩�A�* 

Average reward per step��+�]7��       ��2	L��̩�A�*

epsilon��+�p#Ŏ0       ���_	���̩�A*#
!
Average reward per episode;���Z7��.       ��W�	;��̩�A*!

total reward per episode  �s�{�.       ��W�	C��̩�A�* 

Average reward per step;���ہ��       ��2	��̩�A�*

epsilon;���-`֢.       ��W�	���̩�A�* 

Average reward per step;����>�       ��2	O��̩�A�*

epsilon;���o)'�.       ��W�	����̩�A�* 

Average reward per step;����aq�       ��2	t���̩�A�*

epsilon;���Bkm�.       ��W�	9���̩�A�* 

Average reward per step;�����       ��2	���̩�A�*

epsilon;����"o�.       ��W�	����̩�A�* 

Average reward per step;���wf�y       ��2	[���̩�A�*

epsilon;������y.       ��W�	Ӥ��̩�A�* 

Average reward per step;�����E�       ��2	����̩�A�*

epsilon;����~��.       ��W�	���̩�A�* 

Average reward per step;���9�K�       ��2	����̩�A�*

epsilon;����5��.       ��W�	=���̩�A�* 

Average reward per step;�����I�       ��2	4���̩�A�*

epsilon;�����,.       ��W�	�� �̩�A�* 

Average reward per step;���[Fk       ��2	�� �̩�A�*

epsilon;���P�φ.       ��W�	m��̩�A�* 

Average reward per step;���y2D�       ��2	T��̩�A�*

epsilon;�����.       ��W�	��̩�A�* 

Average reward per step;�����b�       ��2	���̩�A�*

epsilon;���i��X.       ��W�	���̩�A�* 

Average reward per step;�����}�       ��2	O��̩�A�*

epsilon;����PL�.       ��W�	!#�̩�A�* 

Average reward per step;����J�(       ��2	�#�̩�A�*

epsilon;�����,.       ��W�	�T
�̩�A�* 

Average reward per step;���A+�       ��2	~U
�̩�A�*

epsilon;���5��.       ��W�	���̩�A�* 

Average reward per step;����;��       ��2	���̩�A�*

epsilon;���3p�2.       ��W�	؁�̩�A�* 

Average reward per step;���xx~>       ��2	���̩�A�*

epsilon;���`Bkl.       ��W�	���̩�A�* 

Average reward per step;���{�p9       ��2	Ū�̩�A�*

epsilon;����t.       ��W�	��̩�A�* 

Average reward per step;����I�Y       ��2	��̩�A�*

epsilon;���C�1�0       ���_	y"�̩�A*#
!
Average reward per episode�	����.       ��W�	#�̩�A*!

total reward per episode  �c�̐.       ��W�	�S�̩�A�* 

Average reward per step�	����       ��2	�T�̩�A�*

epsilon�	��
	1.       ��W�	�L�̩�A�* 

Average reward per step�	���K�       ��2	fM�̩�A�*

epsilon�	���.       ��W�	�Y�̩�A�* 

Average reward per step�	�-c       ��2	�Z�̩�A�*

epsilon�	��gc.       ��W�	��̩�A�* 

Average reward per step�	�Gˀ       ��2	��̩�A�*

epsilon�	����.       ��W�	��̩�A�* 

Average reward per step�	��r�       ��2	��̩�A�*

epsilon�	� e.       ��W�	.!�̩�A�* 

Average reward per step�	�e�"�       ��2	!�̩�A�*

epsilon�	���y.       ��W�	)x"�̩�A�* 

Average reward per step�	�9�ַ       ��2	�x"�̩�A�*

epsilon�	��s�[.       ��W�	{�$�̩�A�* 

Average reward per step�	�W,��       ��2	]�$�̩�A�*

epsilon�	�����.       ��W�	�'�̩�A�* 

Average reward per step�	�9��       ��2	�'�̩�A�*

epsilon�	����.       ��W�	-�(�̩�A�* 

Average reward per step�	����       ��2	�(�̩�A�*

epsilon�	��M��.       ��W�	'*�̩�A�* 

Average reward per step�	��s�       ��2	�*�̩�A�*

epsilon�	��cs.       ��W�	5�+�̩�A�* 

Average reward per step�	�r#Dk       ��2	0�+�̩�A�*

epsilon�	�cQ��.       ��W�	��-�̩�A�* 

Average reward per step�	����       ��2	E�-�̩�A�*

epsilon�	�D��~.       ��W�	"m0�̩�A�* 

Average reward per step�	�_��[       ��2	�m0�̩�A�*

epsilon�	��	�G.       ��W�	�k2�̩�A�* 

Average reward per step�	�#J�`       ��2	�l2�̩�A�*

epsilon�	�r��>.       ��W�	�#4�̩�A�* 

Average reward per step�	�����       ��2	_$4�̩�A�*

epsilon�	�8��.       ��W�	E)6�̩�A�* 

Average reward per step�	�+H�       ��2	,*6�̩�A�*

epsilon�	�m�W7.       ��W�	)8�̩�A�* 

Average reward per step�	�;�L       ��2	�)8�̩�A�*

epsilon�	�'�C�.       ��W�	�/:�̩�A�* 

Average reward per step�	�LT+H       ��2	s0:�̩�A�*

epsilon�	��G\�.       ��W�	q <�̩�A�* 

Average reward per step�	��q��       ��2	h!<�̩�A�*

epsilon�	���C.       ��W�	��>�̩�A�* 

Average reward per step�	��~�       ��2	��>�̩�A�*

epsilon�	�ʓ%�.       ��W�	�2@�̩�A�* 

Average reward per step�	��j��       ��2	f3@�̩�A�*

epsilon�	�n�D+.       ��W�	�+B�̩�A�* 

Average reward per step�	�q���       ��2	�,B�̩�A�*

epsilon�	���{j.       ��W�	u>D�̩�A�* 

Average reward per step�	���2�       ��2	W?D�̩�A�*

epsilon�	����.       ��W�	N`F�̩�A�* 

Average reward per step�	���s       ��2	aF�̩�A�*

epsilon�	�1�}�.       ��W�	�~H�̩�A�* 

Average reward per step�	�X�Y       ��2	ZH�̩�A�*

epsilon�	����.       ��W�	��J�̩�A�* 

Average reward per step�	��)�       ��2	m�J�̩�A�*

epsilon�	��r�.       ��W�	6�L�̩�A�* 

Average reward per step�	���#       ��2	!�L�̩�A�*

epsilon�	�|���.       ��W�	�N�̩�A�* 

Average reward per step�	��N�
       ��2	|�N�̩�A�*

epsilon�	�,`�.       ��W�	��P�̩�A�* 

Average reward per step�	����d       ��2	��P�̩�A�*

epsilon�	����.       ��W�	�S�̩�A�* 

Average reward per step�	�����       ��2	�S�̩�A�*

epsilon�	��
vT.       ��W�	VFU�̩�A�* 

Average reward per step�	�.�XZ       ��2	�GU�̩�A�*

epsilon�	�Hl��.       ��W�	3�X�̩�A�* 

Average reward per step�	�7<3�       ��2	��X�̩�A�*

epsilon�	���_.       ��W�	
+[�̩�A�* 

Average reward per step�	��T-.       ��2	�+[�̩�A�*

epsilon�	�#��6.       ��W�	I�\�̩�A�* 

Average reward per step�	�k>�c       ��2	��\�̩�A�*

epsilon�	��֔.       ��W�	��_�̩�A�* 

Average reward per step�	�~�9�       ��2	m�_�̩�A�*

epsilon�	�ٚ
%.       ��W�	?a�̩�A�* 

Average reward per step�	���yF       ��2	�a�̩�A�*

epsilon�	�ߺn<.       ��W�	vc�̩�A�* 

Average reward per step�	� `       ��2	!wc�̩�A�*

epsilon�	��ǰ�.       ��W�	
e�̩�A�* 

Average reward per step�	���^�       ��2	�e�̩�A�*

epsilon�	�(�J.       ��W�	(Eg�̩�A�* 

Average reward per step�	�SO       ��2	�Eg�̩�A�*

epsilon�	�be�c.       ��W�	=�i�̩�A�* 

Average reward per step�	��>�       ��2	�i�̩�A�*

epsilon�	�sr.,.       ��W�	S>k�̩�A�* 

Average reward per step�	��U       ��2	�>k�̩�A�*

epsilon�	�a��.       ��W�	U0m�̩�A�* 

Average reward per step�	���w       ��2	+1m�̩�A�*

epsilon�	��6��.       ��W�	�?o�̩�A�* 

Average reward per step�	�]��       ��2	�@o�̩�A�*

epsilon�	��T.       ��W�	M�q�̩�A�* 

Average reward per step�	�j���       ��2	#�q�̩�A�*

epsilon�	�[TO.       ��W�	�@s�̩�A�* 

Average reward per step�	�a�[n       ��2	�As�̩�A�*

epsilon�	���l.       ��W�	ߋu�̩�A�* 

Average reward per step�	���)       ��2	��u�̩�A�*

epsilon�	�U�i$.       ��W�	e�w�̩�A�* 

Average reward per step�	�_�       ��2	�w�̩�A�*

epsilon�	�j#�.       ��W�	C�y�̩�A�* 

Average reward per step�	���       ��2	G�y�̩�A�*

epsilon�	�؆��0       ���_	�z�̩�A*#
!
Average reward per episode�mۿ?���.       ��W�	�z�̩�A*!

total reward per episode  ��vB|.       ��W�	�}�̩�A�* 

Average reward per step�mۿ6C\_       ��2	��}�̩�A�*

epsilon�mۿ�<�.       ��W�	��̩�A�* 

Average reward per step�mۿ0L=Q       ��2	À�̩�A�*

epsilon�mۿ󍮖.       ��W�	�遇̩�A�* 

Average reward per step�mۿ���       ��2	yꁇ̩�A�*

epsilon�mۿ��&�.       ��W�	�僇̩�A�* 

Average reward per step�mۿ$Ojm       ��2	%惇̩�A�*

epsilon�mۿ���.       ��W�	�셇̩�A�* 

Average reward per step�mۿ���       ��2	1텇̩�A�*

epsilon�mۿ��l�.       ��W�	V��̩�A�* 

Average reward per step�mۿ��       ��2	��̩�A�*

epsilon�mۿv���.       ��W�	ظ��̩�A�* 

Average reward per step�mۿS��       ��2	����̩�A�*

epsilon�mۿ3��.       ��W�	z��̩�A�* 

Average reward per step�mۿ���*       ��2	�z��̩�A�*

epsilon�mۿ�=�C.       ��W�	����̩�A�* 

Average reward per step�mۿ3mt`       ��2	����̩�A�*

epsilon�mۿ\�̔.       ��W�	�(��̩�A�* 

Average reward per step�mۿ��O�       ��2	|)��̩�A�*

epsilon�mۿ;2A.       ��W�	���̩�A�* 

Average reward per step�mۿp3��       ��2	���̩�A�*

epsilon�mۿ��M�.       ��W�	-'��̩�A�* 

Average reward per step�mۿ u�       ��2	�'��̩�A�*

epsilon�mۿ��?�.       ��W�	A��̩�A�* 

Average reward per step�mۿ�B�P       ��2	�A��̩�A�*

epsilon�mۿ�ha.       ��W�	�Y��̩�A�* 

Average reward per step�mۿh
��       ��2	�Z��̩�A�*

epsilon�mۿ�b.       ��W�	9���̩�A�* 

Average reward per step�mۿ�	��       ��2	���̩�A�*

epsilon�mۿ =	�.       ��W�	����̩�A�* 

Average reward per step�mۿf��       ��2	����̩�A�*

epsilon�mۿ$C�p.       ��W�	���̩�A�* 

Average reward per step�mۿ��       ��2	ђ��̩�A�*

epsilon�mۿ���.       ��W�	�J��̩�A�* 

Average reward per step�mۿ����       ��2	bK��̩�A�*

epsilon�mۿq��V.       ��W�	��̩�A�* 

Average reward per step�mۿKCC       ��2	���̩�A�*

epsilon�mۿ�`�p.       ��W�	lZ��̩�A�* 

Average reward per step�mۿb�Y       ��2	6[��̩�A�*

epsilon�mۿ> �.       ��W�	3��̩�A�* 

Average reward per step�mۿ�n{       ��2	�3��̩�A�*

epsilon�mۿuw�.       ��W�	����̩�A�* 

Average reward per step�mۿ�q�n       ��2	V���̩�A�*

epsilon�mۿ���q.       ��W�	$��̩�A�* 

Average reward per step�mۿ'��       ��2	���̩�A�*

epsilon�mۿF���.       ��W�	0,��̩�A�* 

Average reward per step�mۿɑ��       ��2	
-��̩�A�*

epsilon�mۿ?��j.       ��W�	�w��̩�A�* 

Average reward per step�mۿ�-O�       ��2	�x��̩�A�*

epsilon�mۿ�0�.       ��W�	���̩�A�* 

Average reward per step�mۿѧ��       ��2	���̩�A�*

epsilon�mۿ��.       ��W�	,I��̩�A�* 

Average reward per step�mۿ�x       ��2	kJ��̩�A�*

epsilon�mۿ�ss�.       ��W�	,~��̩�A�* 

Average reward per step�mۿ�RhP       ��2	��̩�A�*

epsilon�mۿ�Y�0       ���_	�ƹ�̩�A	*#
!
Average reward per episodeI����ܥ).       ��W�	�ǹ�̩�A	*!

total reward per episode  	��n`�.       ��W�	P5��̩�A�* 

Average reward per stepI���A��*       ��2	�5��̩�A�*

epsilonI����vh�.       ��W�	�^��̩�A�* 

Average reward per stepI���׫o�       ��2	�_��̩�A�*

epsilonI�����n.       ��W�	lxÇ̩�A�* 

Average reward per stepI������       ��2	[yÇ̩�A�*

epsilonI���u\�.       ��W�	TŇ̩�A�* 

Average reward per stepI������A       ��2	XŇ̩�A�*

epsilonI���ƪ�.       ��W�	XǇ̩�A�* 

Average reward per stepI���$O{       ��2	�XǇ̩�A�*

epsilonI����I.       ��W�	*�ɇ̩�A�* 

Average reward per stepI���|��       ��2	e�ɇ̩�A�*

epsilonI���~�T�.       ��W�	�ˇ̩�A�* 

Average reward per stepI���M��       ��2	@�ˇ̩�A�*

epsilonI���b-U,.       ��W�	J|͇̩�A�* 

Average reward per stepI����)�`       ��2	5}͇̩�A�*

epsilonI������.       ��W�	%#Ї̩�A�* 

Average reward per stepI������       ��2	�#Ї̩�A�*

epsilonI���2%ʐ.       ��W�	��Ӈ̩�A�* 

Average reward per stepI���H"��       ��2	��Ӈ̩�A�*

epsilonI������0       ���_	��Ӈ̩�A
*#
!
Average reward per episode�����R�.       ��W�	Z�Ӈ̩�A
*!

total reward per episode  '�S��.       ��W�	��ׇ̩�A�* 

Average reward per step����P�Pr       ��2	��ׇ̩�A�*

epsilon������.       ��W�	^�ه̩�A�* 

Average reward per step����{Y�B       ��2	�ه̩�A�*

epsilon����fQ�H.       ��W�	��ۇ̩�A�* 

Average reward per step������       ��2	J�ۇ̩�A�*

epsilon����%��D.       ��W�	uއ̩�A�* 

Average reward per step�����N       ��2	%އ̩�A�*

epsilon�����H�K.       ��W�	B>��̩�A�* 

Average reward per step������`       ��2	)?��̩�A�*

epsilon�����I\.       ��W�	U��̩�A�* 

Average reward per step����+��8       ��2	��̩�A�*

epsilon����X� q.       ��W�	��̩�A�* 

Average reward per step����U��M       ��2	��̩�A�*

epsilon������,a.       ��W�	�Q�̩�A�* 

Average reward per step�������       ��2	�R�̩�A�*

epsilon����-��.       ��W�	���̩�A�* 

Average reward per step�����I��       ��2	���̩�A�*

epsilon�����yx.       ��W�	�F�̩�A�* 

Average reward per step������p�       ��2	�G�̩�A�*

epsilon��������.       ��W�	��̩�A�* 

Average reward per step����fJ�       ��2	��̩�A�*

epsilon�����C�.       ��W�	��̩�A�* 

Average reward per step������3       ��2	��̩�A�*

epsilon�����w�L.       ��W�	3S�̩�A�* 

Average reward per step����E�a+       ��2	T�̩�A�*

epsilon�����T<'.       ��W�	�K�̩�A�* 

Average reward per step�������u       ��2	�L�̩�A�*

epsilon����d2�.       ��W�	_��̩�A�* 

Average reward per step������       ��2	,��̩�A�*

epsilon�����w�Z.       ��W�	ѕ��̩�A�* 

Average reward per step�����F        ��2	���̩�A�*

epsilon�����zW0       ���_	���̩�A*#
!
Average reward per episode  %�k��.       ��W�	����̩�A*!

total reward per episode  %Þww.       ��W�	̲��̩�A�* 

Average reward per step  %�
���       ��2	����̩�A�*

epsilon  %�6�.       ��W�	����̩�A�* 

Average reward per step  %���C�       ��2	p���̩�A�*

epsilon  %�kQ.       ��W�	����̩�A�* 

Average reward per step  %�$�A       ��2	����̩�A�*

epsilon  %���G�.       ��W�	C�̩�A�* 

Average reward per step  %��-�       ��2	�̩�A�*

epsilon  %�ӽ�.       ��W�	�l�̩�A�* 

Average reward per step  %��D�       ��2	n�̩�A�*

epsilon  %���\s.       ��W�	=��̩�A�* 

Average reward per step  %����v       ��2	|��̩�A�*

epsilon  %�r�.       ��W�	�F�̩�A�* 

Average reward per step  %��9��       ��2	�G�̩�A�*

epsilon  %��
�.       ��W�	 T�̩�A�* 

Average reward per step  %�H�m       ��2	U�̩�A�*

epsilon  %��I.       ��W�	��̩�A�* 

Average reward per step  %��l��       ��2	��̩�A�*

epsilon  %�3���.       ��W�	���̩�A�* 

Average reward per step  %�����       ��2	���̩�A�*

epsilon  %����C.       ��W�	`s�̩�A�* 

Average reward per step  %�E5ȶ       ��2	?t�̩�A�*

epsilon  %�wL.       ��W�	��̩�A�* 

Average reward per step  %��*D!       ��2	���̩�A�*

epsilon  %�l��.       ��W�	�Y�̩�A�* 

Average reward per step  %�bШ#       ��2	\Z�̩�A�*

epsilon  %�68.       ��W�	��̩�A�* 

Average reward per step  %���_       ��2	���̩�A�*

epsilon  %����.       ��W�	�0�̩�A�* 

Average reward per step  %�h$�C       ��2	�1�̩�A�*

epsilon  %��ӟE.       ��W�	t�"�̩�A�* 

Average reward per step  %����.       ��2	F�"�̩�A�*

epsilon  %����.       ��W�	H�$�̩�A�* 

Average reward per step  %�'��;       ��2	/�$�̩�A�*

epsilon  %�9ls.       ��W�	I�&�̩�A�* 

Average reward per step  %�,�hv       ��2	�&�̩�A�*

epsilon  %�N 1t.       ��W�	b)�̩�A�* 

Average reward per step  %�vM�       ��2	�)�̩�A�*

epsilon  %�Sήl0       ���_	͓)�̩�A*#
!
Average reward per episode(���c���.       ��W�	ݔ)�̩�A*!

total reward per episode  �l �K.       ��W�	�[/�̩�A�* 

Average reward per step(���<�Li       ��2	�\/�̩�A�*

epsilon(���z�Q*.       ��W�	�m1�̩�A�* 

Average reward per step(����Y��       ��2	an1�̩�A�*

epsilon(������.       ��W�	�95�̩�A�* 

Average reward per step(����l       ��2	d:5�̩�A�*

epsilon(���R���.       ��W�	0�7�̩�A�* 

Average reward per step(����+�       ��2	��7�̩�A�*

epsilon(���]���.       ��W�	�y;�̩�A�* 

Average reward per step(���{���       ��2	�z;�̩�A�*

epsilon(����G�.       ��W�	Ps=�̩�A�* 

Average reward per step(������       ��2	%t=�̩�A�*

epsilon(���?�8.       ��W�	��?�̩�A�* 

Average reward per step(���噣�       ��2	q�?�̩�A�*

epsilon(���
���.       ��W�	��A�̩�A�* 

Average reward per step(���EU�       ��2	��A�̩�A�*

epsilon(���͟�T.       ��W�	�C�̩�A�* 

Average reward per step(������       ��2	��C�̩�A�*

epsilon(������.       ��W�	�E�̩�A�* 

Average reward per step(���B�[�       ��2	%�E�̩�A�*

epsilon(���GQ�.       ��W�	o-H�̩�A�* 

Average reward per step(������O       ��2	j.H�̩�A�*

epsilon(���.k��.       ��W�	��I�̩�A�* 

Average reward per step(�������       ��2	��I�̩�A�*

epsilon(����f�.       ��W�	4�L�̩�A�* 

Average reward per step(����s6�       ��2	
�L�̩�A�*

epsilon(�����߾.       ��W�	"N�̩�A�* 

Average reward per step(���q�y       ��2	#N�̩�A�*

epsilon(�����~.       ��W�	f�O�̩�A�* 

Average reward per step(���P~[]       ��2	��O�̩�A�*

epsilon(���'�C.       ��W�	�=R�̩�A�* 

Average reward per step(����^ȟ       ��2	�>R�̩�A�*

epsilon(�������.       ��W�	0�T�̩�A�* 

Average reward per step(������       ��2	�T�̩�A�*

epsilon(�����^.       ��W�	1xX�̩�A�* 

Average reward per step(����?�7       ��2	�xX�̩�A�*

epsilon(������.       ��W�	�QZ�̩�A�* 

Average reward per step(����ND�       ��2	iRZ�̩�A�*

epsilon(������.       ��W�	yv\�̩�A�* 

Average reward per step(������       ��2	lw\�̩�A�*

epsilon(����G�.       ��W�	>�^�̩�A�* 

Average reward per step(���p��       ��2	��^�̩�A�*

epsilon(���A�S�.       ��W�	Ƥb�̩�A�* 

Average reward per step(���^Gk       ��2	+�b�̩�A�*

epsilon(���)��/.       ��W�	��d�̩�A�* 

Average reward per step(�����Kt       ��2	��d�̩�A�*

epsilon(���?�5.       ��W�	ۉf�̩�A�* 

Average reward per step(���UMD$       ��2	��f�̩�A�*

epsilon(���＠.       ��W�	�h�̩�A�* 

Average reward per step(����Iti       ��2	�h�̩�A�*

epsilon(���?���.       ��W�	��j�̩�A�* 

Average reward per step(���Q���       ��2	ѯj�̩�A�*

epsilon(����"�,0       ���_	��j�̩�A*#
!
Average reward per episode  ��d�.       ��W�	Z�j�̩�A*!

total reward per episode  ï�W�.       ��W�	��n�̩�A�* 

Average reward per step  ��J��       ��2	˝n�̩�A�*

epsilon  ��c/3�.       ��W�	��p�̩�A�* 

Average reward per step  ���u%=       ��2	��p�̩�A�*

epsilon  ��im��.       ��W�	HSs�̩�A�* 

Average reward per step  ��A�О       ��2	�Ts�̩�A�*

epsilon  ��K��l.       ��W�	)u�̩�A�* 

Average reward per step  ���;4       ��2	Ou�̩�A�*

epsilon  �����.       ��W�	��v�̩�A�* 

Average reward per step  ��g?9J       ��2	k�v�̩�A�*

epsilon  ����o.       ��W�	,y�̩�A�* 

Average reward per step  ����o}       ��2	�y�̩�A�*

epsilon  ���L�g.       ��W�	�b{�̩�A�* 

Average reward per step  ��En�^       ��2	kc{�̩�A�*

epsilon  �����.       ��W�	��|�̩�A�* 

Average reward per step  ��Z`��       ��2	��|�̩�A�*

epsilon  �����.       ��W�	�%�̩�A�* 

Average reward per step  ��鑂       ��2	�&�̩�A�*

epsilon  ��6�R`.       ��W�	^���̩�A�* 

Average reward per step  ���t�       ��2	����̩�A�*

epsilon  ��AW0.       ��W�	J���̩�A�* 

Average reward per step  ���w��       ��2	����̩�A�*

epsilon  ��U�7[.       ��W�	�s��̩�A�* 

Average reward per step  ��l
�d       ��2	qt��̩�A�*

epsilon  ���(�.       ��W�	ᗇ�̩�A�* 

Average reward per step  ���n39       ��2	����̩�A�*

epsilon  ��j-%.       ��W�	։�̩�A�* 

Average reward per step  ���	!       ��2	�։�̩�A�*

epsilon  ��"0O.       ��W�	����̩�A�* 

Average reward per step  ��*%�       ��2	p���̩�A�*

epsilon  ���!\i.       ��W�	����̩�A�* 

Average reward per step  ����A�       ��2	eፈ̩�A�*

epsilon  ��M �w.       ��W�	����̩�A�* 

Average reward per step  ��lسY       ��2	x���̩�A�*

epsilon  ��䃿�.       ��W�	�y��̩�A�* 

Average reward per step  ����kO       ��2	�z��̩�A�*

epsilon  ��#� �.       ��W�	�◈̩�A�* 

Average reward per step  ��B�       ��2	�㗈̩�A�*

epsilon  ���JJ�.       ��W�	����̩�A�* 

Average reward per step  ��u33       ��2	w���̩�A�*

epsilon  ��PF�f.       ��W�	����̩�A�* 

Average reward per step  ����_�       ��2	*���̩�A�*

epsilon  ��G���.       ��W�	i��̩�A�* 

Average reward per step  ���I�}       ��2	���̩�A�*

epsilon  ��j:�.       ��W�	����̩�A�* 

Average reward per step  ���\�       ��2	����̩�A�*

epsilon  ���C.       ��W�	���̩�A�* 

Average reward per step  ���̜�       ��2	���̩�A�*

epsilon  ���4��.       ��W�	6��̩�A�* 

Average reward per step  ��\�       ��2	6��̩�A�*

epsilon  ����Ւ.       ��W�	"��̩�A�* 

Average reward per step  �����       ��2	���̩�A�*

epsilon  ���P=�.       ��W�	;㨈̩�A�* 

Average reward per step  ��&�J�       ��2	e䨈̩�A�*

epsilon  ��75d.       ��W�	�V��̩�A�* 

Average reward per step  ����       ��2	�W��̩�A�*

epsilon  ��e��P.       ��W�	�^��̩�A�* 

Average reward per step  ��*H�7       ��2	[_��̩�A�*

epsilon  ����.       ��W�	gd��̩�A�* 

Average reward per step  ��pL�`       ��2	8e��̩�A�*

epsilon  ����F.       ��W�	8���̩�A�* 

Average reward per step  ��tq��       ��2	���̩�A�*

epsilon  ��|	DS.       ��W�	? ��̩�A�* 

Average reward per step  ����       ��2	T��̩�A�*

epsilon  ������.       ��W�	v6��̩�A�* 

Average reward per step  ����       ��2	z7��̩�A�*

epsilon  ��8r.       ��W�	-̩�A�* 

Average reward per step  ���a�       ��2	�̩�A�*

epsilon  ���w��.       ��W�	 ���̩�A�* 

Average reward per step  ��v�1       ��2	0���̩�A�*

epsilon  ������.       ��W�	o־�̩�A�* 

Average reward per step  ���'�       ��2	׾�̩�A�*

epsilon  ������.       ��W�	S���̩�A�* 

Average reward per step  ��۱�       ��2	����̩�A�*

epsilon  ���$�.       ��W�	�!È̩�A�* 

Average reward per step  ��W��
       ��2	�"È̩�A�*

epsilon  ���).       ��W�	�eň̩�A�* 

Average reward per step  ���0�i       ��2	�fň̩�A�*

epsilon  ��YQ��.       ��W�	��ƈ̩�A�* 

Average reward per step  ���F*       ��2	�ƈ̩�A�*

epsilon  ���*��.       ��W�	r�Ȉ̩�A�* 

Average reward per step  ��3�r�       ��2	��Ȉ̩�A�*

epsilon  �����.       ��W�	�Bˈ̩�A�* 

Average reward per step  ���t�Q       ��2	JCˈ̩�A�*

epsilon  ���GAn.       ��W�	w�̩̈�A�* 

Average reward per step  ���uk       ��2	��̩̈�A�*

epsilon  ���5�3.       ��W�	WBψ̩�A�* 

Average reward per step  �����       ��2	FDψ̩�A�*

epsilon  ��S�|.       ��W�	t`ш̩�A�* 

Average reward per step  ���q�       ��2	aш̩�A�*

epsilon  ����*F.       ��W�	gӈ̩�A�* 

Average reward per step  ��Z;       ��2	8hӈ̩�A�*

epsilon  ��Qq,.       ��W�	:XՈ̩�A�* 

Average reward per step  ����L�       ��2	�XՈ̩�A�*

epsilon  �����.       ��W�	d\׈̩�A�* 

Average reward per step  ��.��[       ��2	]׈̩�A�*

epsilon  ���Q�P.       ��W�	�Yو̩�A�* 

Average reward per step  ��X�       ��2	OZو̩�A�*

epsilon  ��ڻ��.       ��W�	�ۈ̩�A�* 

Average reward per step  ��&��       ��2	�ۈ̩�A�*

epsilon  ����&.       ��W�	�݈̩�A�* 

Average reward per step  ���f��       ��2	Φ݈̩�A�*

epsilon  ���X�.       ��W�	��߈̩�A�* 

Average reward per step  �����       ��2	��߈̩�A�*

epsilon  ���Wq�.       ��W�	n��̩�A�* 

Average reward per step  ���CQ       ��2	H��̩�A�*

epsilon  ���f)�.       ��W�	��̩�A�* 

Average reward per step  ���P��       ��2	*��̩�A�*

epsilon  ����b.       ��W�	���̩�A�* 

Average reward per step  ����,�       ��2	���̩�A�*

epsilon  ��	`a.       ��W�	g+�̩�A�* 

Average reward per step  ��b��       ��2	8,�̩�A�*

epsilon  ������.       ��W�	��̩�A�* 

Average reward per step  ����2�       ��2	��̩�A�*

epsilon  ��q���.       ��W�	��̩�A�* 

Average reward per step  ��2�_�       ��2	���̩�A�*

epsilon  ���e.       ��W�	���̩�A�* 

Average reward per step  ����p       ��2	��̩�A�*

epsilon  ����i�.       ��W�	jM�̩�A�* 

Average reward per step  ��M�)       ��2	HN�̩�A�*

epsilon  ��56�.       ��W�	G>��̩�A�* 

Average reward per step  ���|�       ��2	1?��̩�A�*

epsilon  ��+)6�.       ��W�	i:��̩�A�* 

Average reward per step  ��Ǽ       ��2	T;��̩�A�*

epsilon  ����k.       ��W�	E��̩�A�* 

Average reward per step  ��=�m�       ��2	�E��̩�A�*

epsilon  �� ��t.       ��W�	�N��̩�A�* 

Average reward per step  ���晃       ��2	�O��̩�A�*

epsilon  ���L��.       ��W�	�O��̩�A�* 

Average reward per step  ��D� �       ��2	�P��̩�A�*

epsilon  ���S�0       ���_	Gt��̩�A*#
!
Average reward per episode�Nl�m_��.       ��W�	u��̩�A*!

total reward per episode  p��5�.       ��W�	A��̩�A�* 

Average reward per step�Nl���;�       ��2	��̩�A�*

epsilon�Nl���3X.       ��W�	ߨ�̩�A�* 

Average reward per step�Nl�6�       ��2	Ω�̩�A�*

epsilon�Nl�;^z�.       ��W�	���̩�A�* 

Average reward per step�Nl���9>       ��2	���̩�A�*

epsilon�Nl�{!ϳ.       ��W�	��
�̩�A�* 

Average reward per step�Nl��3��       ��2	!�̩�A�*

epsilon�Nl�D��.       ��W�	L��̩�A�* 

Average reward per step�Nl��bNs       ��2	\��̩�A�*

epsilon�Nl���0#.       ��W�	���̩�A�* 

Average reward per step�Nl��ۊ       ��2	͏�̩�A�*

epsilon�Nl���p.       ��W�	\��̩�A�* 

Average reward per step�Nl���ԋ       ��2	!��̩�A�*

epsilon�Nl��� �.       ��W�	�i�̩�A�* 

Average reward per step�Nl���y�       ��2	Yj�̩�A�*

epsilon�Nl���Ii.       ��W�	ZG�̩�A�* 

Average reward per step�Nl��f       ��2	�H�̩�A�*

epsilon�Nl�RIb�.       ��W�	nk�̩�A�* 

Average reward per step�Nl�JU�(       ��2	l�̩�A�*

epsilon�Nl�ҪS�.       ��W�	���̩�A�* 

Average reward per step�Nl�טN       ��2	���̩�A�*

epsilon�Nl�aPt.       ��W�	�d�̩�A�* 

Average reward per step�Nl��@1!       ��2	�e�̩�A�*

epsilon�Nl�EAx.       ��W�	�.$�̩�A�* 

Average reward per step�Nl�t�p       ��2	a2$�̩�A�*

epsilon�Nl��}.       ��W�	*�&�̩�A�* 

Average reward per step�Nl���       ��2	͑&�̩�A�*

epsilon�Nl����K.       ��W�	$((�̩�A�* 

Average reward per step�Nl�}��       ��2	�((�̩�A�*

epsilon�Nl�WG��.       ��W�	�m*�̩�A�* 

Average reward per step�Nl�U\�k       ��2	o*�̩�A�*

epsilon�Nl����.       ��W�	P�,�̩�A�* 

Average reward per step�Nl���[�       ��2	 �,�̩�A�*

epsilon�Nl��2�N.       ��W�	�V.�̩�A�* 

Average reward per step�Nl�C���       ��2	.X.�̩�A�*

epsilon�Nl��3�n.       ��W�	oc0�̩�A�* 

Average reward per step�Nl�]��       ��2	$d0�̩�A�*

epsilon�Nl��!|�.       ��W�	��2�̩�A�* 

Average reward per step�Nl�W��       ��2	ף2�̩�A�*

epsilon�Nl��*0       ���_	�3�̩�A*#
!
Average reward per episode��������.       ��W�	)	3�̩�A*!

total reward per episode  �"\n:.       ��W�	�7�̩�A�* 

Average reward per step�����B��       ��2	�7�̩�A�*

epsilon�����*#.       ��W�	Ec9�̩�A�* 

Average reward per step����R�H�       ��2	�c9�̩�A�*

epsilon����vjV�.       ��W�	�2;�̩�A�* 

Average reward per step�����S�       ��2	z4;�̩�A�*

epsilon�����w�.       ��W�	us=�̩�A�* 

Average reward per step����M��       ��2	�t=�̩�A�*

epsilon�����6�.       ��W�	��?�̩�A�* 

Average reward per step�����v�       ��2	_�?�̩�A�*

epsilon����ɲ��.       ��W�	OWC�̩�A�* 

Average reward per step����WO�       ��2	�XC�̩�A�*

epsilon�����P�.       ��W�	�E�̩�A�* 

Average reward per step����>�8       ��2	��E�̩�A�*

epsilon�����"��.       ��W�	G�̩�A�* 

Average reward per step����G�e	       ��2	�G�̩�A�*

epsilon������.       ��W�	�sI�̩�A�* 

Average reward per step����t��       ��2	�tI�̩�A�*

epsilon�����r.       ��W�	��K�̩�A�* 

Average reward per step����&}�|       ��2	@�K�̩�A�*

epsilon����h��.       ��W�	۾M�̩�A�* 

Average reward per step�����b��       ��2	��M�̩�A�*

epsilon����=�K@.       ��W�	UO�̩�A�* 

Average reward per step����(L��       ��2	�UO�̩�A�*

epsilon�����Vp.       ��W�	ԷQ�̩�A�* 

Average reward per step����e@�       ��2	��Q�̩�A�*

epsilon�����\�.       ��W�	 T�̩�A�* 

Average reward per step�����q       ��2	� T�̩�A�*

epsilon����[�n.       ��W�	6�U�̩�A�* 

Average reward per step��������       ��2	%�U�̩�A�*

epsilon����v&0�.       ��W�	�PW�̩�A�* 

Average reward per step����� �       ��2	�QW�̩�A�*

epsilon����et�.       ��W�	 �Y�̩�A�* 

Average reward per step����#��       ��2	��Y�̩�A�*

epsilon����́�.       ��W�	�[�̩�A�* 

Average reward per step����G-�^       ��2	�[�̩�A�*

epsilon�����U.       ��W�	�
^�̩�A�* 

Average reward per step������kZ       ��2	x^�̩�A�*

epsilon����o� �.       ��W�	5`�̩�A�* 

Average reward per step�����3[�       ��2	�5`�̩�A�*

epsilon����5H�.       ��W�	��a�̩�A�* 

Average reward per step����Jo�       ��2	�a�̩�A�*

epsilon������.       ��W�	�,d�̩�A�* 

Average reward per step�����j��       ��2	<.d�̩�A�*

epsilon����1�9.       ��W�	^cf�̩�A�* 

Average reward per step����g}O�       ��2	�df�̩�A�*

epsilon����!�L�.       ��W�	�h�̩�A�* 

Average reward per step�������F       ��2	� h�̩�A�*

epsilon�����B�.       ��W�	�~j�̩�A�* 

Average reward per step����.�{�       ��2	Vj�̩�A�*

epsilon������}m.       ��W�	��l�̩�A�* 

Average reward per step�����zz	       ��2	��l�̩�A�*

epsilon����Qs��.       ��W�	5bn�̩�A�* 

Average reward per step������V,       ��2	�cn�̩�A�*

epsilon����\h3+.       ��W�	6�p�̩�A�* 

Average reward per step����v���       ��2	�p�̩�A�*

epsilon�����kw.       ��W�	��r�̩�A�* 

Average reward per step�����"��       ��2	"�r�̩�A�*

epsilon�������.       ��W�	v�t�̩�A�* 

Average reward per step����"�G       ��2	H�t�̩�A�*

epsilon����_ �.       ��W�	s�v�̩�A�* 

Average reward per step����i�r�       ��2	Q�v�̩�A�*

epsilon�����_�.       ��W�	�fy�̩�A�* 

Average reward per step�����*f9       ��2	<hy�̩�A�*

epsilon����5��.       ��W�	�;}�̩�A�* 

Average reward per step����G_ca       ��2	�<}�̩�A�*

epsilon�������.       ��W�	+�~�̩�A�* 

Average reward per step����0*�/       ��2	�~�̩�A�*

epsilon����q�x4.       ��W�	�N��̩�A�* 

Average reward per step�������       ��2	"P��̩�A�*

epsilon����gmaa0       ���_	�s��̩�A*#
!
Average reward per episodeuP�z><o.       ��W�	�t��̩�A*!

total reward per episode  �ɼV.       ��W�	�K��̩�A�* 

Average reward per stepuP����       ��2	@M��̩�A�*

epsilonuP�R��.       ��W�	˂��̩�A�* 

Average reward per stepuP��u��       ��2	����̩�A�*

epsilonuP�b <.       ��W�	G ��̩�A�* 

Average reward per stepuP���d       ��2	���̩�A�*

epsilonuP��X"p.       ��W�	a��̩�A�* 

Average reward per stepuP�?���       ��2	�a��̩�A�*

epsilonuP�u��G.       ��W�	!���̩�A�* 

Average reward per stepuP�{L�       ��2	`���̩�A�*

epsilonuP����.       ��W�	�ӑ�̩�A�* 

Average reward per stepuP�����       ��2	kԑ�̩�A�*

epsilonuP��@A�.       ��W�	�&��̩�A�* 

Average reward per stepuP�~Bg�       ��2	�'��̩�A�*

epsilonuP�XA}�.       ��W�	~ŕ�̩�A�* 

Average reward per stepuP��}ҥ       ��2	.ƕ�̩�A�*

epsilonuP�M�.       ��W�	q=��̩�A�* 

Average reward per stepuP�}r�       ��2	:>��̩�A�*

epsilonuP���u�.       ��W�	~���̩�A�* 

Average reward per stepuP��sm�       ��2	X���̩�A�*

epsilonuP�Ͱ�.       ��W�	��̩�A�* 

Average reward per stepuP�"89�       ��2	A�̩�A�*

epsilonuP�Q_�^.       ��W�	����̩�A�* 

Average reward per stepuP��%6�       ��2	x���̩�A�*

epsilonuP��k"=.       ��W�	2 ��̩�A�* 

Average reward per stepuP�(�j{       ��2	y��̩�A�*

epsilonuP���..       ��W�	����̩�A�* 

Average reward per stepuP���{       ��2	Ԟ��̩�A�*

epsilonuP�NF�.       ��W�	���̩�A�* 

Average reward per stepuP�g�U�       ��2	]���̩�A�*

epsilonuP�MN�H.       ��W�	z4��̩�A�* 

Average reward per stepuP����6       ��2	35��̩�A�*

epsilonuP�W�8{.       ��W�	�N��̩�A�* 

Average reward per stepuP����M       ��2	�O��̩�A�*

epsilonuP� �"P.       ��W�	����̩�A�* 

Average reward per stepuP���       ��2	ۊ��̩�A�*

epsilonuP��&�.       ��W�	yX��̩�A�* 

Average reward per stepuP���}       ��2	Z��̩�A�*

epsilonuP�O�$.       ��W�	����̩�A�* 

Average reward per stepuP��A       ��2	ǻ��̩�A�*

epsilonuP�hֲ�.       ��W�	���̩�A�* 

Average reward per stepuP�i�(�       ��2	����̩�A�*

epsilonuP�AS.       ��W�	x_��̩�A�* 

Average reward per stepuP���D�       ��2	`��̩�A�*

epsilonuP�X��G.       ��W�	R���̩�A�* 

Average reward per stepuP�^ǌ�       ��2	w���̩�A�*

epsilonuP�W��.       ��W�	mY��̩�A�* 

Average reward per stepuP�-��v       ��2	�Z��̩�A�*

epsilonuP��-h�.       ��W�	)о�̩�A�* 

Average reward per stepuP�'��       ��2	�о�̩�A�*

epsilonuP���\.       ��W�	~��̩�A�* 

Average reward per stepuP��	�       ��2	\��̩�A�*

epsilonuP��y�.       ��W�	��̩�A�* 

Average reward per stepuP�����       ��2	w�̩�A�*

epsilonuP�nG%n.       ��W�		ŉ̩�A�* 

Average reward per stepuP���5       ��2	(
ŉ̩�A�*

epsilonuP�0	(.       ��W�	��Ɖ̩�A�* 

Average reward per stepuP�~z�       ��2	t�Ɖ̩�A�*

epsilonuP���3.       ��W�	�	ɉ̩�A�* 

Average reward per stepuP��+.       ��2	t
ɉ̩�A�*

epsilonuP�tͤc.       ��W�	��ʉ̩�A�* 

Average reward per stepuP���       ��2	��ʉ̩�A�*

epsilonuP�U~.       ��W�	W͉̩�A�* 

Average reward per stepuP�5�G       ��2	p͉̩�A�*

epsilonuP�(�B.       ��W�	zRω̩�A�* 

Average reward per stepuP��f��       ��2	iSω̩�A�*

epsilonuP���.       ��W�	�+щ̩�A�* 

Average reward per stepuP��ۨ       ��2	�,щ̩�A�*

epsilonuP�ު��.       ��W�	Q�Ӊ̩�A�* 

Average reward per stepuP���[�       ��2	j�Ӊ̩�A�*

epsilonuP�qU�e.       ��W�	#I׉̩�A�* 

Average reward per stepuP�21�X       ��2	^J׉̩�A�*

epsilonuP��<�K.       ��W�	l�ى̩�A�* 

Average reward per stepuP�@q        ��2	W�ى̩�A�*

epsilonuP�����.       ��W�	3P̩݉�A�* 

Average reward per stepuP�̫c�       ��2		Q̩݉�A�*

epsilonuP�т��0       ���_	v̩݉�A*#
!
Average reward per episode(�A��X�L.       ��W�	w̩݉�A*!

total reward per episode  �����.       ��W�	_a�̩�A�* 

Average reward per step(�A�;���       ��2	5b�̩�A�*

epsilon(�A��ܱ{.       ��W�	��̩�A�* 

Average reward per step(�A���;�       ��2	ޯ�̩�A�*

epsilon(�A�.cֆ.       ��W�	���̩�A�* 

Average reward per step(�A��2       ��2	���̩�A�*

epsilon(�A�E{��.       ��W�	��̩�A�* 

Average reward per step(�A�t�f       ��2	��̩�A�*

epsilon(�A����.       ��W�	���̩�A�* 

Average reward per step(�A�8;Q�       ��2	F��̩�A�*

epsilon(�A��V5.       ��W�	�A�̩�A�* 

Average reward per step(�A�X:��       ��2	�B�̩�A�*

epsilon(�A�U"��.       ��W�	g��̩�A�* 

Average reward per step(�A���Z       ��2	V��̩�A�*

epsilon(�A���.       ��W�	'�̩�A�* 

Average reward per step(�A����M       ��2	�'�̩�A�*

epsilon(�A��!Uf.       ��W�	���̩�A�* 

Average reward per step(�A�"�       ��2	���̩�A�*

epsilon(�A�l�X.       ��W�	���̩�A�* 

Average reward per step(�A��F       ��2	���̩�A�*

epsilon(�A��Z��.       ��W�	�I��̩�A�* 

Average reward per step(�A��G�       ��2	�J��̩�A�*

epsilon(�A��GmE.       ��W�	���̩�A�* 

Average reward per step(�A� Kf       ��2	���̩�A�*

epsilon(�A�{��.       ��W�	�!��̩�A�* 

Average reward per step(�A���#       ��2	�"��̩�A�*

epsilon(�A�'�S.       ��W�	5y��̩�A�* 

Average reward per step(�A�*�^�       ��2	Jz��̩�A�*

epsilon(�A���.       ��W�	����̩�A�* 

Average reward per step(�A�'o       ��2	  �̩�A�*

epsilon(�A�n,'[.       ��W�	ni�̩�A�* 

Average reward per step(�A�2���       ��2	wj�̩�A�*

epsilon(�A���"7.       ��W�	���̩�A�* 

Average reward per step(�A���       ��2	ݚ�̩�A�*

epsilon(�A�I�.       ��W�	cG�̩�A�* 

Average reward per step(�A��k��       ��2	�H�̩�A�*

epsilon(�A�F�Uw.       ��W�	��̩�A�* 

Average reward per step(�A�|tE�       ��2	��̩�A�*

epsilon(�A��$�.       ��W�	 �
�̩�A�* 

Average reward per step(�A�_C��       ��2	�
�̩�A�*

epsilon(�A��/�.       ��W�	���̩�A�* 

Average reward per step(�A�s�}�       ��2	/��̩�A�*

epsilon(�A��ٸ�.       ��W�	���̩�A�* 

Average reward per step(�A�c��       ��2	  �̩�A�*

epsilon(�A�0��.       ��W�	���̩�A�* 

Average reward per step(�A��}$       ��2	İ�̩�A�*

epsilon(�A�al�".       ��W�	V��̩�A�* 

Average reward per step(�A��=�n       ��2	���̩�A�*

epsilon(�A�uTh.       ��W�	���̩�A�* 

Average reward per step(�A� �       ��2	/��̩�A�*

epsilon(�A�-��.       ��W�	���̩�A�* 

Average reward per step(�A���
       ��2	���̩�A�*

epsilon(�A���sS.       ��W�	Z�̩�A�* 

Average reward per step(�A��lC�       ��2	F[�̩�A�*

epsilon(�A��V#�.       ��W�	��̩�A�* 

Average reward per step(�A�ٍ>       ��2	��̩�A�*

epsilon(�A�6�0�.       ��W�	�n�̩�A�* 

Average reward per step(�A�w)�       ��2	;o�̩�A�*

epsilon(�A���F�.       ��W�	�`!�̩�A�* 

Average reward per step(�A��lI�       ��2	si!�̩�A�*

epsilon(�A�r���.       ��W�	�#�̩�A�* 

Average reward per step(�A�U9Q&       ��2	��#�̩�A�*

epsilon(�A���`@.       ��W�	�q%�̩�A�* 

Average reward per step(�A�{��       ��2	�r%�̩�A�*

epsilon(�A�QXA.       ��W�	'�̩�A�* 

Average reward per step(�A��K       ��2	'�̩�A�*

epsilon(�A�aJh.       ��W�	�B*�̩�A�* 

Average reward per step(�A�!9��       ��2	�C*�̩�A�*

epsilon(�A�IRi.       ��W�	��-�̩�A�* 

Average reward per step(�A�jSR�       ��2	��-�̩�A�*

epsilon(�A���Xo.       ��W�	(-0�̩�A�* 

Average reward per step(�A���#�       ��2	#.0�̩�A�*

epsilon(�A�8�t.       ��W�	S4�̩�A�* 

Average reward per step(�A�W�@%       ��2	}4�̩�A�*

epsilon(�A�	F2.       ��W�	��5�̩�A�* 

Average reward per step(�A��}��       ��2	פ5�̩�A�*

epsilon(�A�߿��.       ��W�	��7�̩�A�* 

Average reward per step(�A�g���       ��2	��7�̩�A�*

epsilon(�A�̀ό.       ��W�	�U:�̩�A�* 

Average reward per step(�A�4�<�       ��2	~V:�̩�A�*

epsilon(�A�9�'.       ��W�	q�<�̩�A�* 

Average reward per step(�A��=��       ��2	��<�̩�A�*

epsilon(�A����L.       ��W�	]3@�̩�A�* 

Average reward per step(�A��qB       ��2	z4@�̩�A�*

epsilon(�A����Z.       ��W�	AbB�̩�A�* 

Average reward per step(�A�WF�       ��2	�cB�̩�A�*

epsilon(�A�tM�7.       ��W�	��C�̩�A�* 

Average reward per step(�A��sfq       ��2	��C�̩�A�*

epsilon(�A��zPH.       ��W�	itF�̩�A�* 

Average reward per step(�A��\\1       ��2	6uF�̩�A�*

epsilon(�A�-/W�.       ��W�	H�̩�A�* 

Average reward per step(�A���ND       ��2	~H�̩�A�*

epsilon(�A�fd]�.       ��W�	�hJ�̩�A�* 

Average reward per step(�A�eː-       ��2	jJ�̩�A�*

epsilon(�A��f�#.       ��W�	V�L�̩�A�* 

Average reward per step(�A��?�       ��2	0�L�̩�A�*

epsilon(�A�c%3?0       ���_	��L�̩�A*#
!
Average reward per episode�������.       ��W�	Z�L�̩�A*!

total reward per episode  ��7�6�.       ��W�	*R�̩�A�* 

Average reward per step���]�l�       ��2	(+R�̩�A�*

epsilon���dz�9.       ��W�	�T�̩�A�* 

Average reward per step���P�t�       ��2	 �T�̩�A�*

epsilon������.       ��W�	�qV�̩�A�* 

Average reward per step����':       ��2	TrV�̩�A�*

epsilon���A���.       ��W�	�/X�̩�A�* 

Average reward per step���FD       ��2	Q1X�̩�A�*

epsilon����a.�.       ��W�	W%[�̩�A�* 

Average reward per step���#��o       ��2	-&[�̩�A�*

epsilon���;a�.       ��W�	��\�̩�A�* 

Average reward per step����k*m       ��2	ǻ\�̩�A�*

epsilon����[A�.       ��W�	�^�̩�A�* 

Average reward per step������       ��2	Έ^�̩�A�*

epsilon���b�%.       ��W�	��`�̩�A�* 

Average reward per step����띻       ��2	˼`�̩�A�*

epsilon������Y.       ��W�	R~c�̩�A�* 

Average reward per step�����5�       ��2	c�̩�A�*

epsilon���.�nO.       ��W�	�lf�̩�A�* 

Average reward per step���5��       ��2	�mf�̩�A�*

epsilon���@1��.       ��W�	��g�̩�A�* 

Average reward per step���H��       ��2	��g�̩�A�*

epsilon���}7R�.       ��W�	�j�̩�A�* 

Average reward per step�����       ��2	ˁj�̩�A�*

epsilon���"��.       ��W�	�k�̩�A�* 

Average reward per step���ؠ!�       ��2	��k�̩�A�*

epsilon����#�.       ��W�	� n�̩�A�* 

Average reward per step����t�       ��2	B!n�̩�A�*

epsilon����c�.       ��W�	�lp�̩�A�* 

Average reward per step���1#�D       ��2	amp�̩�A�*

epsilon���P��.       ��W�	�r�̩�A�* 

Average reward per step���z�c�       ��2	 
r�̩�A�*

epsilon���� yb.       ��W�	'�t�̩�A�* 

Average reward per step����Y O       ��2	��t�̩�A�*

epsilon����p��.       ��W�	�Kv�̩�A�* 

Average reward per step����3�:       ��2	�Lv�̩�A�*

epsilon���n��.       ��W�	N�x�̩�A�* 

Average reward per step������       ��2	Z�x�̩�A�*

epsilon����]A�.       ��W�	! z�̩�A�* 

Average reward per step����Ⱦ�       ��2	� z�̩�A�*

epsilon����6MV.       ��W�	�6|�̩�A�* 

Average reward per step����g1�       ��2	X7|�̩�A�*

epsilon����\�Z.       ��W�	S�~�̩�A�* 

Average reward per step����#�       ��2	)�~�̩�A�*

epsilon���Z��0       ���_	x�~�̩�A*#
!
Average reward per episode]t��� �.       ��W�	k�~�̩�A*!

total reward per episode  �I�.       ��W�	�R��̩�A�* 

Average reward per step]t��E���       ��2	vS��̩�A�*

epsilon]t��v�dN.       ��W�	�x��̩�A�* 

Average reward per step]t������       ��2	ty��̩�A�*

epsilon]t��rK��.       ��W�	���̩�A�* 

Average reward per step]t���[ߘ       ��2	M���̩�A�*

epsilon]t��?�T�.       ��W�	?��̩�A�* 

Average reward per step]t��)��F       ��2	W@��̩�A�*

epsilon]t��r`�.       ��W�	�̌�̩�A�* 

Average reward per step]t��ę�/       ��2	Ό�̩�A�*

epsilon]t����{�.       ��W�	o��̩�A�* 

Average reward per step]t��d3�       ��2	+��̩�A�*

epsilon]t����Rq.       ��W�	x��̩�A�* 

Average reward per step]t��zf�       ��2	A��̩�A�*

epsilon]t���2eA.       ��W�	g��̩�A�* 

Average reward per step]t��ŕ��       ��2	E��̩�A�*

epsilon]t��栱S.       ��W�	UL��̩�A�* 

Average reward per step]t���5o       ��2	�L��̩�A�*

epsilon]t��(%.       ��W�	�(��̩�A�* 

Average reward per step]t���r�       ��2	�)��̩�A�*

epsilon]t��BG�.       ��W�	�q��̩�A�* 

Average reward per step]t��3N`       ��2	�r��̩�A�*

epsilon]t������.       ��W�	{���̩�A�* 

Average reward per step]t���	��       ��2	���̩�A�*

epsilon]t��S4.       ��W�	
��̩�A�* 

Average reward per step]t��Ѕm       ��2	��̩�A�*

epsilon]t��1�L.       ��W�	���̩�A�* 

Average reward per step]t���'�       ��2	����̩�A�*

epsilon]t���U.       ��W�	6ʥ�̩�A�* 

Average reward per step]t����       ��2	�˥�̩�A�*

epsilon]t��u�I.       ��W�	#���̩�A�* 

Average reward per step]t���WW       ��2	؟��̩�A�*

epsilon]t����n).       ��W�	�U��̩�A�* 

Average reward per step]t��g��       ��2	�V��̩�A�*

epsilon]t��ǃ�.       ��W�	P���̩�A�* 

Average reward per step]t���i�       ��2	T���̩�A�*

epsilon]t����}w.       ��W�	`?��̩�A�* 

Average reward per step]t���w�i       ��2	!@��̩�A�*

epsilon]t����S�.       ��W�	Z���̩�A�* 

Average reward per step]t���9       ��2	(���̩�A�*

epsilon]t���|�{.       ��W�	����̩�A�* 

Average reward per step]t��!P��       ��2	N���̩�A�*

epsilon]t��o�Bn.       ��W�	�#��̩�A�* 

Average reward per step]t��?Z��       ��2	�$��̩�A�*

epsilon]t���Mg.       ��W�	Ӣ��̩�A�* 

Average reward per step]t���K�A       ��2	����̩�A�*

epsilon]t���s[.       ��W�	����̩�A�* 

Average reward per step]t����s�       ��2	���̩�A�*

epsilon]t��Ĝj.       ��W�	�̺�̩�A�* 

Average reward per step]t��a5/�       ��2	�ͺ�̩�A�*

epsilon]t����Y.       ��W�	����̩�A�* 

Average reward per step]t���rg�       ��2	�̩�A�*

epsilon]t���"g.       ��W�	5���̩�A�* 

Average reward per step]t��&|�       ��2	����̩�A�*

epsilon]t��#���.       ��W�	a���̩�A�* 

Average reward per step]t��#R	       ��2	D���̩�A�*

epsilon]t���$�,.       ��W�	��Ê̩�A�* 

Average reward per step]t��)^R�       ��2	��Ê̩�A�*

epsilon]t���AKS.       ��W�	�Ǌ̩�A�* 

Average reward per step]t�����       ��2	oǊ̩�A�*

epsilon]t�����U.       ��W�	�#Ɋ̩�A�* 

Average reward per step]t��J)2U       ��2	�$Ɋ̩�A�*

epsilon]t���y��.       ��W�	�ʊ̩�A�* 

Average reward per step]t��8�8�       ��2	�ʊ̩�A�*

epsilon]t��j3�.       ��W�	3m̩͊�A�* 

Average reward per step]t������       ��2	�m̩͊�A�*

epsilon]t��&V�r.       ��W�	.vϊ̩�A�* 

Average reward per step]t��+H�?       ��2	�vϊ̩�A�*

epsilon]t��G-1�.       ��W�	ъ̩�A�* 

Average reward per step]t��G;Fy       ��2	�ъ̩�A�*

epsilon]t��%!eT.       ��W�	e�Ҋ̩�A�* 

Average reward per step]t��A+�h       ��2	~�Ҋ̩�A�*

epsilon]t����.       ��W�	mՊ̩�A�* 

Average reward per step]t��v�Tj       ��2	6Պ̩�A�*

epsilon]t���ܕ.       ��W�	5�׊̩�A�* 

Average reward per step]t��
S�$       ��2	ճ׊̩�A�*

epsilon]t��Q.��.       ��W�	
ي̩�A�* 

Average reward per step]t������       ��2	�ي̩�A�*

epsilon]t���	~.       ��W�	�ۊ̩�A�* 

Average reward per step]t��R���       ��2	ɬۊ̩�A�*

epsilon]t���".       ��W�	3̩݊�A�* 

Average reward per step]t���҅       ��2	�̩݊�A�*

epsilon]t���[�.       ��W�	ߊ̩�A�* 

Average reward per step]t����c�       ��2	�ߊ̩�A�*

epsilon]t��d4�v.       ��W�	:@�̩�A�* 

Average reward per step]t���ƃ       ��2	�@�̩�A�*

epsilon]t�����.       ��W�	�Q�̩�A�* 

Average reward per step]t����r�       ��2	/R�̩�A�*

epsilon]t��p��K.       ��W�	�[�̩�A�* 

Average reward per step]t��VGV       ��2	J]�̩�A�*

epsilon]t������.       ��W�	��̩�A�* 

Average reward per step]t�����4       ��2	���̩�A�*

epsilon]t���n5.       ��W�	`��̩�A�* 

Average reward per step]t��1��J       ��2	S��̩�A�*

epsilon]t��-ռ>.       ��W�	��̩�A�* 

Average reward per step]t��,J�       ��2	���̩�A�*

epsilon]t��;4�.       ��W�	��̩�A�* 

Average reward per step]t���M�       ��2	���̩�A�*

epsilon]t����ʂ0       ���_	�̩�A*#
!
Average reward per episode����p��.       ��W�	��̩�A*!

total reward per episode  ����.       ��W�	i��̩�A�* 

Average reward per step����*�       ��2	��̩�A�*

epsilon����o.       ��W�	ٖ��̩�A�* 

Average reward per step�����x(u       ��2	����̩�A�*

epsilon�����y�.       ��W�	����̩�A�* 

Average reward per step��������       ��2	����̩�A�*

epsilon����L�%R.       ��W�	����̩�A�* 

Average reward per step����ς$�       ��2	����̩�A�*

epsilon�����F�.       ��W�	���̩�A�* 

Average reward per step����o��       ��2	ٕ��̩�A�*

epsilon�����A�.       ��W�	����̩�A�* 

Average reward per step�����8��       ��2	����̩�A�*

epsilon�����Y�.       ��W�	����̩�A�* 

Average reward per step����\��h       ��2	����̩�A�*

epsilon����w�.       ��W�	2� �̩�A�* 

Average reward per step����L�"�       ��2	ͫ �̩�A�*

epsilon�������*.       ��W�	.�̩�A�* 

Average reward per step����R�%       ��2	�̩�A�*

epsilon������-�.       ��W�	5��̩�A�* 

Average reward per step�����<0�       ��2	
��̩�A�*

epsilon�������.       ��W�	��̩�A�* 

Average reward per step�����(�       ��2	j��̩�A�*

epsilon����:��w.       ��W�	 ��̩�A�* 

Average reward per step����=��       ��2	���̩�A�*

epsilon������.       ��W�	D4	�̩�A�* 

Average reward per step����3e�%       ��2	5	�̩�A�*

epsilon������0.       ��W�	�5�̩�A�* 

Average reward per step����#3�       ��2	D6�̩�A�*

epsilon�����?G�.       ��W�	]7�̩�A�* 

Average reward per step�����m�       ��2	78�̩�A�*

epsilon����Ө�.       ��W�	
.�̩�A�* 

Average reward per step�����Ty�       ��2	�.�̩�A�*

epsilon����B��/.       ��W�	�%�̩�A�* 

Average reward per step����A�o�       ��2	R&�̩�A�*

epsilon�����N��.       ��W�	��̩�A�* 

Average reward per step����HD,       ��2	��̩�A�*

epsilon����
�AD.       ��W�	3P�̩�A�* 

Average reward per step����`5<       ��2	�P�̩�A�*

epsilon��������.       ��W�	ep�̩�A�* 

Average reward per step����_H&�       ��2	�p�̩�A�*

epsilon����M�B`.       ��W�	ρ�̩�A�* 

Average reward per step������w       ��2	w��̩�A�*

epsilon����gs] .       ��W�	ϣ�̩�A�* 

Average reward per step�����C!�       ��2	f��̩�A�*

epsilon�������0       ���_	s��̩�A*#
!
Average reward per episode�E�����\.       ��W�	��̩�A*!

total reward per episode  ��4�.       ��W�	PU�̩�A�* 

Average reward per step�E��r��C       ��2	!V�̩�A�*

epsilon�E��ٖI�.       ��W�	�c �̩�A�* 

Average reward per step�E��
��       ��2	kd �̩�A�*

epsilon�E�����*.       ��W�	W["�̩�A�* 

Average reward per step�E���%�       ��2	�["�̩�A�*

epsilon�E��<huI.       ��W�	8�$�̩�A�* 

Average reward per step�E��v�Y       ��2	܂$�̩�A�*

epsilon�E��~�.       ��W�	�&�̩�A�* 

Average reward per step�E���v�       ��2	�&�̩�A�*

epsilon�E��1�a~.       ��W�	��(�̩�A�* 

Average reward per step�E����@       ��2	��(�̩�A�*

epsilon