       �K"	  �X̩�Abrain.Event:2^�T���      �*	�X�X̩�A"��
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
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
Index0*
T0*
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
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�~ٽ
_
dense_1/random_uniform/maxConst*
valueB
 *�~�=*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�*
seed2���*
seed���)
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
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	�
�
dense_1/kernel
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*!
_class
loc:@dense_1/kernel
|
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�
\
dense_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
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
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
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
dense_2/random_uniform/shapeConst*
_output_shapes
:*
valueB"   d   *
dtype0
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
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2���*
seed���)
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
T0*
_output_shapes
: 
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes
:	�d
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
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:d*
T0
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
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:d2*
T0
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
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:d2
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
VariableV2*
_output_shapes
:2*
	container *
shape:2*
shared_name *
dtype0
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
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
_output_shapes

:2*
seed2��K*
seed���)
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
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
�
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:2
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
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
dense_5/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�m?*
dtype0
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
dtype0*
_output_shapes

:*
seed2��*
seed���)*
T0
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 
�
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
_output_shapes

:*
T0
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes

:
�
dense_5/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
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
dense_5/kernel/readIdentitydense_5/kernel*
T0*!
_class
loc:@dense_5/kernel*
_output_shapes

:
Z
dense_5/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
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
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
o
lambda_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"       
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
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*
T0*'
_output_shapes
:���������
o
lambda_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"       
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
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
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
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
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
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: 
j
Adam/beta_1/readIdentityAdam/beta_1*
_class
loc:@Adam/beta_1*
_output_shapes
: *
T0
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
Adam/beta_2/readIdentityAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
T0
]
Adam/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
n

Adam/decay
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
|
flatten_1_input_1Placeholder* 
shape:���������*
dtype0*+
_output_shapes
:���������
b
flatten_1_1/ShapeShapeflatten_1_input_1*
_output_shapes
:*
T0*
out_type0
i
flatten_1_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
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
shrink_axis_mask *
ellipsis_mask *

begin_mask *
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
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�*
seed2��h*
seed���)
�
dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
_output_shapes
: *
T0
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
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name 
�
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*
_output_shapes
:	�*
use_locking(*
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(
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
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
data_formatNHWC*(
_output_shapes
:����������*
T0
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
dense_2_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?��=
�
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes
:	�d*
seed2Ӽ�
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
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
_output_shapes
:	�d*
use_locking(*
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(
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
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*!
_class
loc:@dense_2_1/bias
w
dense_2_1/bias/readIdentitydense_2_1/bias*
_output_shapes
:d*
T0*!
_class
loc:@dense_2_1/bias
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
dtype0*
_output_shapes
:*
valueB"d   2   
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
VariableV2*
shape
:d2*
shared_name *
dtype0*
_output_shapes

:d2*
	container 
�
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking(
�
dense_3_1/kernel/readIdentitydense_3_1/kernel*
T0*#
_class
loc:@dense_3_1/kernel*
_output_shapes

:d2
\
dense_3_1/ConstConst*
dtype0*
_output_shapes
:2*
valueB2*    
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
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
_output_shapes
:2*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(
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
_output_shapes

:2*
seed2ֹM*
seed���)*
T0*
dtype0
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
dense_4_1/bias/readIdentitydense_4_1/bias*
T0*!
_class
loc:@dense_4_1/bias*
_output_shapes
:
�
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
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
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2��*
seed���)
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
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
_output_shapes

:*
T0
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
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@dense_5_1/kernel
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
�
dense_5_1/bias/AssignAssigndense_5_1/biasdense_5_1/Const*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
w
dense_5_1/bias/readIdentitydense_5_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_5_1/bias
�
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
lambda_1_1/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
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
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*#
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
d
lambda_1_1/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
Index0*
T0
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
lambda_1_1/strided_slice_2StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
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
dtype0*
_output_shapes
: *
_class
loc:@dense_1/bias
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
IsVariableInitialized_11IsVariableInitializedAdam/lr*
dtype0*
_output_shapes
: *
_class
loc:@Adam/lr
�
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1
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
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*
_output_shapes
: *#
_class
loc:@dense_5_1/kernel*
dtype0
�
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*
_output_shapes
: *!
_class
loc:@dense_5_1/bias*
dtype0
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
shape:	�*
dtype0*
_output_shapes
:	�
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
Placeholder_1Placeholder*
shape:�*
dtype0*
_output_shapes	
:�
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
Placeholder_2Placeholder*
dtype0*
_output_shapes
:	�d*
shape:	�d
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
dtype0*
_output_shapes

:d2*
shape
:d2
�
Assign_4Assigndense_3_1/kernelPlaceholder_4*
_output_shapes

:d2*
use_locking( *
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(
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
Placeholder_7Placeholder*
shape:*
dtype0*
_output_shapes
:
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
Placeholder_9Placeholder*
shape:*
dtype0*
_output_shapes
:
�
Assign_9Assigndense_5_1/biasPlaceholder_9*
validate_shape(*
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@dense_5_1/bias
^
SGD/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
r
SGD/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
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
SGD/momentum/readIdentitySGD/momentum*
_output_shapes
: *
T0*
_class
loc:@SGD/momentum
\
SGD/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
m
	SGD/decay
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: 
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
lambda_1_sample_weightsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
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
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*#
_output_shapes
:���������*
T0
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

loss/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
T0*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: *
use_locking(
a
SGD_1/lr/readIdentitySGD_1/lr*
_class
loc:@SGD_1/lr*
_output_shapes
: *
T0
a
SGD_1/momentum/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
r
SGD_1/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*!
_class
loc:@SGD_1/momentum*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
_class
loc:@SGD_1/decay*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
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
loss_1/lambda_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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

loss_1/mulMulloss_1/mul/xloss_1/lambda_1_loss/Mean_3*
_output_shapes
: *
T0
i
y_truePlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
g
maskPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
Y

loss_2/subSublambda_1/suby_true*'
_output_shapes
:���������*
T0
O

loss_2/AbsAbs
loss_2/sub*'
_output_shapes
:���������*
T0
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
loss_2/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
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
loss_2/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
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
loss_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
t
lambda_1_sample_weights_2Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
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
loss_3/loss_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:���������
b
loss_3/loss_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss_3/mul/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*#
_output_shapes
:���������*
T0
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
loss_3/lambda_1_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*
T0*'
_output_shapes
:���������

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
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
metrics_2/mean_q/ConstConst*
_output_shapes
:*
valueB: *
dtype0
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
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
IsVariableInitialized_25IsVariableInitializedSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
y
IsVariableInitialized_26IsVariableInitializedSGD/lr*
_output_shapes
: *
_class
loc:@SGD/lr*
dtype0
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
IsVariableInitialized_29IsVariableInitializedSGD_1/iterations*
dtype0	*
_output_shapes
: *#
_class
loc:@SGD_1/iterations
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
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"�qYpF      ��w	���X̩�AJ��
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
shrink_axis_mask *

begin_mask *
ellipsis_mask *
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
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
\
flatten_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
t
flatten_1/stackPackflatten_1/stack/0flatten_1/Prod*

axis *
N*
_output_shapes
:*
T0
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
dense_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�~ٽ
_
dense_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�~�=*
dtype0
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
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes
:	�*
T0
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
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
:	�*
T0
\
dense_1/ConstConst*
valueB�*    *
dtype0*
_output_shapes	
:�
z
dense_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:�*
	container *
shape:�
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
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:����������*
transpose_a( 
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
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�d*
seed2���*
seed���)
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
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�d
�
dense_2/kernel
VariableV2*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name *
dtype0
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
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
:	�d*
T0
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
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:d*
use_locking(
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:d*
T0
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
_output_shapes

:d2*
seed2�ځ*
seed���)*
T0*
dtype0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
�
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:d2*
T0
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:d2
�
dense_3/kernel
VariableV2*
shape
:d2*
shared_name *
dtype0*
_output_shapes

:d2*
	container 
�
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:d2
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
VariableV2*
_output_shapes
:2*
	container *
shape:2*
shared_name *
dtype0
�
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
_output_shapes
:2*
use_locking(*
T0*
_class
loc:@dense_3/bias*
validate_shape(
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
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*'
_output_shapes
:���������2*
T0*
data_formatNHWC
\
activation_3/ReluReludense_3/BiasAdd*'
_output_shapes
:���������2*
T0
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
dense_4/random_uniform/maxConst*
valueB
 *�D�>*
dtype0*
_output_shapes
: 
�
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
_output_shapes

:2*
seed2��K*
seed���)
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
�
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:2
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
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:2
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
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(
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
dense_5/random_uniform/minConst*
_output_shapes
: *
valueB
 *�m�*
dtype0
_
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�m?
�
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:*
seed2��
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
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
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
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
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
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
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
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
o
lambda_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
q
 lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
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
dtype0*
_output_shapes
:*
valueB"      
�
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0
_
lambda_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
lambda_1/MeanMeanlambda_1/strided_slice_2lambda_1/Const*
T0*
_output_shapes

:*

Tidx0*
	keep_dims(
b
lambda_1/subSublambda_1/addlambda_1/Mean*'
_output_shapes
:���������*
T0
_
Adam/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
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
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
_output_shapes
: *
valueB
 *o�9*
dtype0
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
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/lr
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*
dtype0
o
Adam/beta_2
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
flatten_1_1/ShapeShapeflatten_1_input_1*
out_type0*
_output_shapes
:*
T0
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
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask 
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
flatten_1_1/stack/0Const*
valueB :
���������*
dtype0*
_output_shapes
: 
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
N*
_output_shapes
:*
T0*

axis 
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
_output_shapes
:	�*
seed2��h*
seed���)*
T0*
dtype0
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
dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
T0*
_output_shapes
:	�
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
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*#
_class
loc:@dense_1_1/kernel
�
dense_1_1/kernel/readIdentitydense_1_1/kernel*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	�*
T0
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
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*!
_class
loc:@dense_1_1/bias
x
dense_1_1/bias/readIdentitydense_1_1/bias*
T0*!
_class
loc:@dense_1_1/bias*
_output_shapes	
:�
�
dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*(
_output_shapes
:����������*
transpose_a( *
transpose_b( *
T0
�
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*(
_output_shapes
:����������*
T0*
data_formatNHWC
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
dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
_output_shapes
:	�d*
T0
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
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
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
dense_3_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
�
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2���*
seed���)
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
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
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
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2*
use_locking(
w
dense_3_1/bias/readIdentitydense_3_1/bias*
_output_shapes
:2*
T0*!
_class
loc:@dense_3_1/bias
�
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( *
T0
�
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*'
_output_shapes
:���������2*
T0*
data_formatNHWC
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
dense_4_1/random_uniform/mulMul&dense_4_1/random_uniform/RandomUniformdense_4_1/random_uniform/sub*
_output_shapes

:2*
T0
�
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
T0*
_output_shapes

:2
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
dense_4_1/kernel/readIdentitydense_4_1/kernel*
T0*#
_class
loc:@dense_4_1/kernel*
_output_shapes

:2
\
dense_4_1/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
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
dense_4_1/bias/AssignAssigndense_4_1/biasdense_4_1/Const*
T0*!
_class
loc:@dense_4_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
w
dense_4_1/bias/readIdentitydense_4_1/bias*
_output_shapes
:*
T0*!
_class
loc:@dense_4_1/bias
�
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
o
dense_5_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_5_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *�m�*
dtype0
a
dense_5_1/random_uniform/maxConst*
valueB
 *�m?*
dtype0*
_output_shapes
: 
�
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
_output_shapes

:*
seed2��*
seed���)*
T0*
dtype0
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
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
dense_5_1/kernel/readIdentitydense_5_1/kernel*
_output_shapes

:*
T0*#
_class
loc:@dense_5_1/kernel
\
dense_5_1/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
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
dense_5_1/bias/AssignAssigndense_5_1/biasdense_5_1/Const*
use_locking(*
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:
w
dense_5_1/bias/readIdentitydense_5_1/bias*!
_class
loc:@dense_5_1/bias*
_output_shapes
:*
T0
�
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
o
lambda_1_1/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
q
 lambda_1_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
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
"lambda_1_1/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*'
_output_shapes
:���������*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
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
"lambda_1_1/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
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
lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*
T0*
_output_shapes

:*

Tidx0*
	keep_dims(
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*'
_output_shapes
:���������*
T0
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_output_shapes
: *
_class
loc:@dense_1/bias*
dtype0
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
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_output_shapes
: *
_class
loc:@dense_3/bias*
dtype0
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
IsVariableInitialized_11IsVariableInitializedAdam/lr*
_output_shapes
: *
_class
loc:@Adam/lr*
dtype0
�
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_2
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
IsVariableInitialized_19IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_20IsVariableInitializeddense_3_1/bias*
dtype0*
_output_shapes
: *!
_class
loc:@dense_3_1/bias
�
IsVariableInitialized_21IsVariableInitializeddense_4_1/kernel*
dtype0*
_output_shapes
: *#
_class
loc:@dense_4_1/kernel
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
PlaceholderPlaceholder*
shape:	�*
dtype0*
_output_shapes
:	�
�
AssignAssigndense_1_1/kernelPlaceholder*
validate_shape(*
_output_shapes
:	�*
use_locking( *
T0*#
_class
loc:@dense_1_1/kernel
X
Placeholder_1Placeholder*
shape:�*
dtype0*
_output_shapes	
:�
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
Placeholder_2Placeholder*
dtype0*
_output_shapes
:	�d*
shape:	�d
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
Placeholder_5Placeholder*
_output_shapes
:2*
shape:2*
dtype0
�
Assign_5Assigndense_3_1/biasPlaceholder_5*
validate_shape(*
_output_shapes
:2*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias
^
Placeholder_6Placeholder*
_output_shapes

:2*
shape
:2*
dtype0
�
Assign_6Assigndense_4_1/kernelPlaceholder_6*
validate_shape(*
_output_shapes

:2*
use_locking( *
T0*#
_class
loc:@dense_4_1/kernel
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
Assign_8Assigndense_5_1/kernelPlaceholder_8*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking( 
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:*
shape:
�
Assign_9Assigndense_5_1/biasPlaceholder_9*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:*
use_locking( *
T0
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
r
SGD/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB 
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
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD_1/iterations/AssignAssignSGD_1/iterationsSGD_1/iterations/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*#
_class
loc:@SGD_1/iterations
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
SGD_1/momentum/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
r
SGD_1/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
SGD_1/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
T0*
_class
loc:@SGD_1/decay*
validate_shape(*
_output_shapes
: *
use_locking(
j
SGD_1/decay/readIdentitySGD_1/decay*
T0*
_class
loc:@SGD_1/decay*
_output_shapes
: 
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
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
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
loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*#
_output_shapes
:���������*
T0
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
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:���������
f
loss_1/lambda_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
loss_2/Less/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
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
loss_2/mul_2Mulloss_2/Selectmask*
T0*'
_output_shapes
:���������
g
loss_2/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
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
loss_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:���������*
shape:���������
t
lambda_1_sample_weights_2Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
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
loss_3/loss_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
loss_3/lambda_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
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
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
`
metrics_2/mean_q/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
metrics_2/mean_q/MeanMeanmetrics_2/mean_q/Maxmetrics_2/mean_q/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
[
metrics_2/mean_q/Const_1Const*
_output_shapes
: *
valueB *
dtype0
�
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
IsVariableInitialized_29IsVariableInitializedSGD_1/iterations*
dtype0	*
_output_shapes
: *#
_class
loc:@SGD_1/iterations
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
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0�%�.       ��W�	���Y̩�A*#
!
Average reward per episodew��4�|,       ���E	w��Y̩�A*!

total reward per episode  �\��-       <A��	`��Y̩�A+* 

Average reward per stepw��9�7c       `/�#	!��Y̩�A+*

epsilonw����:�-       <A��	��Y̩�A,* 

Average reward per stepw����-�       `/�#	���Y̩�A,*

epsilonw���1�-       <A��	��Y̩�A-* 

Average reward per stepw�����       `/�#	u��Y̩�A-*

epsilonw��TMF-       <A��	���Y̩�A.* 

Average reward per stepw��J��6       `/�#	&��Y̩�A.*

epsilonw���ׂ�-       <A��	���Y̩�A/* 

Average reward per stepw��y��       `/�#	v��Y̩�A/*

epsilonw������-       <A��	���Y̩�A0* 

Average reward per stepw���}�       `/�#	���Y̩�A0*

epsilonw���~�-       <A��	���Y̩�A1* 

Average reward per stepw��st"�       `/�#	���Y̩�A1*

epsilonw��Mn�|-       <A��	Ϊ�Y̩�A2* 

Average reward per stepw��9��       `/�#	`��Y̩�A2*

epsilonw��ìHJ-       <A��	���Y̩�A3* 

Average reward per stepw���>�       `/�#	���Y̩�A3*

epsilonw��W���-       <A��	+��Y̩�A4* 

Average reward per stepw�����R       `/�#	���Y̩�A4*

epsilonw��1��-       <A��	l��Y̩�A5* 

Average reward per stepw���n��       `/�#	B��Y̩�A5*

epsilonw���9�F-       <A��	���Y̩�A6* 

Average reward per stepw�����       `/�#	\��Y̩�A6*

epsilonw��r��R-       <A��	e��Y̩�A7* 

Average reward per stepw���VC       `/�#	��Y̩�A7*

epsilonw���,.3-       <A��	���Y̩�A8* 

Average reward per stepw���}�`       `/�#	���Y̩�A8*

epsilonw�����-       <A��	v��Y̩�A9* 

Average reward per stepw��xb       `/�#	L��Y̩�A9*

epsilonw��^���-       <A��	���Y̩�A:* 

Average reward per stepw��??�       `/�#	���Y̩�A:*

epsilonw���e��-       <A��	��Y̩�A;* 

Average reward per stepw����T       `/�#	���Y̩�A;*

epsilonw��vw=�-       <A��	>��Y̩�A<* 

Average reward per stepw�����m       `/�#	���Y̩�A<*

epsilonw��ɋ &-       <A��	J��Y̩�A=* 

Average reward per stepw��sn�b       `/�#	���Y̩�A=*

epsilonw����[�-       <A��	
�Y̩�A>* 

Average reward per stepw��&� J       `/�#	�
�Y̩�A>*

epsilonw��L��-       <A��	G��Y̩�A?* 

Average reward per stepw��G8�	       `/�#	���Y̩�A?*

epsilonw���"!0       ���_	���Y̩�A*#
!
Average reward per episodez���/�|�.       ��W�	C��Y̩�A*!

total reward per episode  �3vä-       <A��	��Y̩�A@* 

Average reward per stepz����3{       `/�#	v�Y̩�A@*

epsilonz���*b�-       <A��	���Y̩�AA* 

Average reward per stepz���
�       `/�#	]��Y̩�AA*

epsilonz�����u�-       <A��	J�Y̩�AB* 

Average reward per stepz���kHc�       `/�#	K�Y̩�AB*

epsilonz����>"�-       <A��	y;�Y̩�AC* 

Average reward per stepz����KZ       `/�#	:<�Y̩�AC*

epsilonz��� ���-       <A��	�1�Y̩�AD* 

Average reward per stepz�����       `/�#	�2�Y̩�AD*

epsilonz����#?4-       <A��	$�Y̩�AE* 

Average reward per stepz�������       `/�#	��Y̩�AE*

epsilonz�����)�-       <A��	1A�Y̩�AF* 

Average reward per stepz���)&X9       `/�#	�A�Y̩�AF*

epsilonz���N���-       <A��	� Z̩�AG* 

Average reward per stepz����yd3       `/�#	~ Z̩�AG*

epsilonz����q^�-       <A��	@KZ̩�AH* 

Average reward per stepz����H�6       `/�#	�KZ̩�AH*

epsilonz����vX�-       <A��	N*Z̩�AI* 

Average reward per stepz���T��       `/�#	�*Z̩�AI*

epsilonz�����L�-       <A��	�Z̩�AJ* 

Average reward per stepz����t|�       `/�#	eZ̩�AJ*

epsilonz�������-       <A��	�ZZ̩�AK* 

Average reward per stepz���|Y%0       `/�#	�[Z̩�AK*

epsilonz���
�o-       <A��	&9Z̩�AL* 

Average reward per stepz����p�       `/�#	�9Z̩�AL*

epsilonz������}-       <A��	x	Z̩�AM* 

Average reward per stepz������C       `/�#	�x	Z̩�AM*

epsilonz������-       <A��	�gZ̩�AN* 

Average reward per stepz���"l�       `/�#	QhZ̩�AN*

epsilonz���#�-       <A��	�@Z̩�AO* 

Average reward per stepz������       `/�#	�AZ̩�AO*

epsilonz����3��-       <A��	�vZ̩�AP* 

Average reward per stepz����#:�       `/�#	uwZ̩�AP*

epsilonz����&|J-       <A��	�Z̩�AQ* 

Average reward per stepz�����F�       `/�#	aZ̩�AQ*

epsilonz��� A0       ���_	�1Z̩�A*#
!
Average reward per episode  �����.       ��W�	f2Z̩�A*!

total reward per episode  +��u�-       <A��	�PZ̩�AR* 

Average reward per step  ����       `/�#	RZ̩�AR*

epsilon  ��b�-       <A��	�Z̩�AS* 

Average reward per step  �9$�       `/�#	��Z̩�AS*

epsilon  ����X-       <A��	�nZ̩�AT* 

Average reward per step  ��4��       `/�#	�oZ̩�AT*

epsilon  ��N�-       <A��	�[Z̩�AU* 

Average reward per step  ��V�v       `/�#	�\Z̩�AU*

epsilon  �[�>�-       <A��	Y�Z̩�AV* 

Average reward per step  �'�a�       `/�#	#�Z̩�AV*

epsilon  ���w�-       <A��	�Z̩�AW* 

Average reward per step  �ӷ��       `/�#	�Z̩�AW*

epsilon  �^Zj-       <A��	�_Z̩�AX* 

Average reward per step  �o��       `/�#	[`Z̩�AX*

epsilon  ��F�c-       <A��	P�Z̩�AY* 

Average reward per step  ��X�       `/�#	��Z̩�AY*

epsilon  ��0�{-       <A��	�f!Z̩�AZ* 

Average reward per step  ���       `/�#	g!Z̩�AZ*

epsilon  �EC�-       <A��	��"Z̩�A[* 

Average reward per step  �ݘP�       `/�#	I�"Z̩�A[*

epsilon  �%�W-       <A��	�~$Z̩�A\* 

Average reward per step  ��_ߥ       `/�#	E$Z̩�A\*

epsilon  �#�{�-       <A��	÷%Z̩�A]* 

Average reward per step  ��Gy-       `/�#	c�%Z̩�A]*

epsilon  �0|r-       <A��	��'Z̩�A^* 

Average reward per step  ��"�       `/�#	��'Z̩�A^*

epsilon  �C�1-       <A��	.�(Z̩�A_* 

Average reward per step  ��ՊQ       `/�#	 �(Z̩�A_*

epsilon  �jPA-       <A��	ɬ*Z̩�A`* 

Average reward per step  �Q{J       `/�#	��*Z̩�A`*

epsilon  �U���-       <A��	g�,Z̩�Aa* 

Average reward per step  �N��       `/�#	E�,Z̩�Aa*

epsilon  ��^_2-       <A��	��-Z̩�Ab* 

Average reward per step  �q���       `/�#	�-Z̩�Ab*

epsilon  ��<(-       <A��	��/Z̩�Ac* 

Average reward per step  �YAz       `/�#	Q�/Z̩�Ac*

epsilon  �;���-       <A��	��0Z̩�Ad* 

Average reward per step  ���k       `/�#	��0Z̩�Ad*

epsilon  �A���-       <A��	�2Z̩�Ae* 

Average reward per step  �T��       `/�#	��2Z̩�Ae*

epsilon  ��b-       <A��	��4Z̩�Af* 

Average reward per step  ��H6\       `/�#	��4Z̩�Af*

epsilon  ����-       <A��	T�5Z̩�Ag* 

Average reward per step  �lݢ       `/�#	�5Z̩�Ag*

epsilon  ��)�-       <A��	��7Z̩�Ah* 

Average reward per step  ��u       `/�#	��7Z̩�Ah*

epsilon  �NS��-       <A��	��9Z̩�Ai* 

Average reward per step  �~��       `/�#	;�9Z̩�Ai*

epsilon  ��u--       <A��	u�:Z̩�Aj* 

Average reward per step  ��?��       `/�#	�:Z̩�Aj*

epsilon  �*yr�-       <A��	H�<Z̩�Ak* 

Average reward per step  �Ó�N       `/�#	�<Z̩�Ak*

epsilon  ��9]-       <A��	��=Z̩�Al* 

Average reward per step  ��~4U       `/�#	��=Z̩�Al*

epsilon  �9׃s-       <A��	�?Z̩�Am* 

Average reward per step  �x:L        `/�#	��?Z̩�Am*

epsilon  ��0
-       <A��	k�AZ̩�An* 

Average reward per step  �a*�       `/�#	��AZ̩�An*

epsilon  �0�޶-       <A��	V�BZ̩�Ao* 

Average reward per step  �\��V       `/�#	�BZ̩�Ao*

epsilon  �6��-       <A��	:�DZ̩�Ap* 

Average reward per step  ���       `/�#	��DZ̩�Ap*

epsilon  ��3^�-       <A��	&FZ̩�Aq* 

Average reward per step  ����       `/�#	�FZ̩�Aq*

epsilon  ���p�-       <A��	��GZ̩�Ar* 

Average reward per step  ���t�       `/�#	a�GZ̩�Ar*

epsilon  ��1�-       <A��	jIZ̩�As* 

Average reward per step  ��pS       `/�#	IZ̩�As*

epsilon  ���-       <A��	�
KZ̩�At* 

Average reward per step  �)Gۙ       `/�#	VKZ̩�At*

epsilon  �~�Ja-       <A��	L�LZ̩�Au* 

Average reward per step  ��	>�       `/�#	��LZ̩�Au*

epsilon  �{�kU-       <A��	0�NZ̩�Av* 

Average reward per step  ��S�!       `/�#	��NZ̩�Av*

epsilon  ��y%-       <A��	
PZ̩�Aw* 

Average reward per step  ��cF       `/�#	�PZ̩�Aw*

epsilon  ���]Y-       <A��	�QZ̩�Ax* 

Average reward per step  ��hd*       `/�#	��QZ̩�Ax*

epsilon  �e���-       <A��	�SZ̩�Ay* 

Average reward per step  ��._�       `/�#	� SZ̩�Ay*

epsilon  ��$q-       <A��	5UZ̩�Az* 

Average reward per step  ����e       `/�#	OUZ̩�Az*

epsilon  ��d~p-       <A��	�VZ̩�A{* 

Average reward per step  ��O�       `/�#	H�VZ̩�A{*

epsilon  �����-       <A��	?5XZ̩�A|* 

Average reward per step  ��ϸ       `/�#	6XZ̩�A|*

epsilon  ��=�-       <A��	aZZ̩�A}* 

Average reward per step  ��^�       `/�#	�ZZ̩�A}*

epsilon  ����3-       <A��	��[Z̩�A~* 

Average reward per step  ��o��       `/�#	f�[Z̩�A~*

epsilon  ��M4�-       <A��	2]Z̩�A* 

Average reward per step  ����       `/�#	�2]Z̩�A*

epsilon  ��=�.       ��W�	_Z̩�A�* 

Average reward per step  �u�b       ��2	�_Z̩�A�*

epsilon  ��S].       ��W�	|D`Z̩�A�* 

Average reward per step  �Fm��       ��2	E`Z̩�A�*

epsilon  ��/�.       ��W�	�&bZ̩�A�* 

Average reward per step  �_�\�       ��2	�'bZ̩�A�*

epsilon  �d��.       ��W�	�2dZ̩�A�* 

Average reward per step  �wՋo       ��2	f3dZ̩�A�*

epsilon  �I�?�.       ��W�	�+fZ̩�A�* 

Average reward per step  �!��~       ��2	�,fZ̩�A�*

epsilon  �K��.       ��W�	ygZ̩�A�* 

Average reward per step  �	�s       ��2	�ygZ̩�A�*

epsilon  ��J�.       ��W�	jiZ̩�A�* 

Average reward per step  �v���       ��2	�jiZ̩�A�*

epsilon  �*�χ.       ��W�	j�kZ̩�A�* 

Average reward per step  �����       ��2	�kZ̩�A�*

epsilon  ��8��.       ��W�	)�mZ̩�A�* 

Average reward per step  ��w�6       ��2	дmZ̩�A�*

epsilon  �M���.       ��W�	x�oZ̩�A�* 

Average reward per step  �7K{       ��2	 �oZ̩�A�*

epsilon  ����.       ��W�	��qZ̩�A�* 

Average reward per step  ��`       ��2	n�qZ̩�A�*

epsilon  �($��.       ��W�	̴sZ̩�A�* 

Average reward per step  �6C��       ��2	��sZ̩�A�*

epsilon  �e��*.       ��W�	�uZ̩�A�* 

Average reward per step  �\	Z�       ��2	�uZ̩�A�*

epsilon  �zȐ�.       ��W�	�wZ̩�A�* 

Average reward per step  ��ղ=       ��2	��wZ̩�A�*

epsilon  �C�27.       ��W�	��yZ̩�A�* 

Average reward per step  ���a       ��2	^�yZ̩�A�*

epsilon  ���;.       ��W�	�{Z̩�A�* 

Average reward per step  �N%��       ��2	�{Z̩�A�*

epsilon  �|��.       ��W�	3�|Z̩�A�* 

Average reward per step  �h�
�       ��2	7�|Z̩�A�*

epsilon  ��4s�.       ��W�	Օ~Z̩�A�* 

Average reward per step  �4��       ��2	��~Z̩�A�*

epsilon  ��(T.       ��W�	W�Z̩�A�* 

Average reward per step  ����       ��2	(�Z̩�A�*

epsilon  ��x\T.       ��W�	*�Z̩�A�* 

Average reward per step  �R��K       ��2	�Z̩�A�*

epsilon  �Ė�j.       ��W�	I��Z̩�A�* 

Average reward per step  �m\k       ��2	��Z̩�A�*

epsilon  �J$g.       ��W�	:�Z̩�A�* 

Average reward per step  ��j�p       ��2	-�Z̩�A�*

epsilon  ��z�.       ��W�	���Z̩�A�* 

Average reward per step  �қk       ��2	��Z̩�A�*

epsilon  �+sS�.       ��W�	�C�Z̩�A�* 

Average reward per step  �����       ��2	�D�Z̩�A�*

epsilon  �-�\�.       ��W�		O�Z̩�A�* 

Average reward per step  �����       ��2	�O�Z̩�A�*

epsilon  �p�4�.       ��W�	<�Z̩�A�* 

Average reward per step  �Iv�       ��2	�=�Z̩�A�*

epsilon  �̀:.       ��W�	u:�Z̩�A�* 

Average reward per step  �����       ��2	>=�Z̩�A�*

epsilon  ��.       ��W�	WA�Z̩�A�* 

Average reward per step  ����       ��2	SB�Z̩�A�*

epsilon  ��3
�.       ��W�	 _�Z̩�A�* 

Average reward per step  ��汝       ��2	�_�Z̩�A�*

epsilon  �����0       ���_	}z�Z̩�A*#
!
Average reward per episode)\��a�/.       ��W�	{�Z̩�A*!

total reward per episode  (��6�.       ��W�	�#�Z̩�A�* 

Average reward per step)\��ܤ�       ��2	�$�Z̩�A�*

epsilon)\����o.       ��W�	��Z̩�A�* 

Average reward per step)\�L"ݻ       ��2	��Z̩�A�*

epsilon)\��$	�.       ��W�	��Z̩�A�* 

Average reward per step)\��� �       ��2	��Z̩�A�*

epsilon)\��fr�.       ��W�	W]�Z̩�A�* 

Average reward per step)\���r       ��2	5^�Z̩�A�*

epsilon)\���.       ��W�	Ab�Z̩�A�* 

Average reward per step)\����#       ��2	 c�Z̩�A�*

epsilon)\�@5�p.       ��W�	�l�Z̩�A�* 

Average reward per step)\�q��       ��2	am�Z̩�A�*

epsilon)\����.       ��W�	�`�Z̩�A�* 

Average reward per step)\�{,r�       ��2	�a�Z̩�A�*

epsilon)\���n�.       ��W�	iU�Z̩�A�* 

Average reward per step)\���E       ��2	eV�Z̩�A�*

epsilon)\�*J.       ��W�	x��Z̩�A�* 

Average reward per step)\�rs�       ��2	E��Z̩�A�*

epsilon)\��x�?.       ��W�	<��Z̩�A�* 

Average reward per step)\����       ��2	���Z̩�A�*

epsilon)\�U�W�.       ��W�	iǬZ̩�A�* 

Average reward per step)\��T�       ��2	TȬZ̩�A�*

epsilon)\�s�-�.       ��W�	f��Z̩�A�* 

Average reward per step)\��Т       ��2	/��Z̩�A�*

epsilon)\�F�S�.       ��W�	!�Z̩�A�* 

Average reward per step)\����f       ��2	��Z̩�A�*

epsilon)\�3�Ž.       ��W�	�εZ̩�A�* 

Average reward per step)\�#D��       ��2	�ϵZ̩�A�*

epsilon)\��D�.       ��W�	�Q�Z̩�A�* 

Average reward per step)\���       ��2	�R�Z̩�A�*

epsilon)\��[�.       ��W�	�G�Z̩�A�* 

Average reward per step)\��L]0       ��2	�H�Z̩�A�*

epsilon)\�zBO.       ��W�	V~�Z̩�A�* 

Average reward per step)\�wV�A       ��2	E��Z̩�A�*

epsilon)\�p9�i.       ��W�	ھZ̩�A�* 

Average reward per step)\��Rb       ��2	�ھZ̩�A�*

epsilon)\�4�U�.       ��W�	�6�Z̩�A�* 

Average reward per step)\�;n�       ��2	�7�Z̩�A�*

epsilon)\�`z�.       ��W�	���Z̩�A�* 

Average reward per step)\�=��       ��2	���Z̩�A�*

epsilon)\�h7?�.       ��W�	��Z̩�A�* 

Average reward per step)\��W��       ��2	C�Z̩�A�*

epsilon)\���3�.       ��W�	��Z̩�A�* 

Average reward per step)\���0T       ��2	��Z̩�A�*

epsilon)\��f2i.       ��W�	vO�Z̩�A�* 

Average reward per step)\�6��       ��2	YP�Z̩�A�*

epsilon)\�2��.       ��W�	�a�Z̩�A�* 

Average reward per step)\�Jcښ       ��2	|b�Z̩�A�*

epsilon)\�Ę��.       ��W�	��Z̩�A�* 

Average reward per step)\���f&       ��2	���Z̩�A�*

epsilon)\�u9�l0       ���_	��Z̩�A*#
!
Average reward per episode\���O�H�.       ��W�	��Z̩�A*!

total reward per episode  ����k.       ��W�	IM�Z̩�A�* 

Average reward per step\���v��       ��2	N�Z̩�A�*

epsilon\�����KV.       ��W�	�?�Z̩�A�* 

Average reward per step\���/ʕ       ��2	l@�Z̩�A�*

epsilon\����(�.       ��W�	a6�Z̩�A�* 

Average reward per step\���2�("       ��2	77�Z̩�A�*

epsilon\���(��%.       ��W�	σ�Z̩�A�* 

Average reward per step\����FG8       ��2	���Z̩�A�*

epsilon\�������.       ��W�	z��Z̩�A�* 

Average reward per step\����B��       ��2	��Z̩�A�*

epsilon\���J�.       ��W�	V~�Z̩�A�* 

Average reward per step\���yU�        ��2	�~�Z̩�A�*

epsilon\����M�m.       ��W�	�v�Z̩�A�* 

Average reward per step\���$�V�       ��2	�w�Z̩�A�*

epsilon\����k|�.       ��W�	b��Z̩�A�* 

Average reward per step\���"��       ��2	8��Z̩�A�*

epsilon\�����.       ��W�	?��Z̩�A�* 

Average reward per step\����g`�       ��2	��Z̩�A�*

epsilon\������l.       ��W�	��Z̩�A�* 

Average reward per step\����CQ       ��2	ܷ�Z̩�A�*

epsilon\�������.       ��W�	H��Z̩�A�* 

Average reward per step\���	�p       ��2	"��Z̩�A�*

epsilon\���F��.       ��W�	:��Z̩�A�* 

Average reward per step\���3n^�       ��2	���Z̩�A�*

epsilon\���DQ�.       ��W�	�Z̩�A�* 

Average reward per step\���w��t       ��2	��Z̩�A�*

epsilon\���PC1l.       ��W�	6�Z̩�A�* 

Average reward per step\����>b       ��2	�Z̩�A�*

epsilon\���m��.       ��W�	���Z̩�A�* 

Average reward per step\����7��       ��2	���Z̩�A�*

epsilon\���g��.       ��W�	,��Z̩�A�* 

Average reward per step\���I��       ��2	0��Z̩�A�*

epsilon\���m��.       ��W�	��Z̩�A�* 

Average reward per step\����\�R       ��2	��Z̩�A�*

epsilon\����9��.       ��W�	<�Z̩�A�* 

Average reward per step\����Y�R       ��2	��Z̩�A�*

epsilon\�����".       ��W�	B&�Z̩�A�* 

Average reward per step\����<�       ��2	k'�Z̩�A�*

epsilon\���u�׼.       ��W�	�:�Z̩�A�* 

Average reward per step\���L��       ��2	�;�Z̩�A�*

epsilon\���I��.       ��W�	p?�Z̩�A�* 

Average reward per step\����h�_       ��2	J@�Z̩�A�*

epsilon\��� �u).       ��W�	�c�Z̩�A�* 

Average reward per step\�����~       ��2	Nd�Z̩�A�*

epsilon\�����M.       ��W�	���Z̩�A�* 

Average reward per step\���ϴ5�       ��2	���Z̩�A�*

epsilon\���Z�y�.       ��W�	\��Z̩�A�* 

Average reward per step\���i���       ��2	���Z̩�A�*

epsilon\�����l�.       ��W�	��Z̩�A�* 

Average reward per step\����Ց�       ��2	���Z̩�A�*

epsilon\���>�oW.       ��W�	���Z̩�A�* 

Average reward per step\���N
U       ��2	���Z̩�A�*

epsilon\�����j.       ��W�	 [̩�A�* 

Average reward per step\���Sb�       ��2	� [̩�A�*

epsilon\����}�p.       ��W�	Nb[̩�A�* 

Average reward per step\���?NI@       ��2	Ac[̩�A�*

epsilon\���	�'.       ��W�	�Y[̩�A�* 

Average reward per step\������       ��2	\Z[̩�A�*

epsilon\������.       ��W�	d[̩�A�* 

Average reward per step\���I���       ��2	�d[̩�A�*

epsilon\���B7��.       ��W�	�T[̩�A�* 

Average reward per step\���yϴ3       ��2	�U[̩�A�*

epsilon\���<���.       ��W�	5B	[̩�A�* 

Average reward per step\���|n9�       ��2	�B	[̩�A�*

epsilon\����R��.       ��W�	
�
[̩�A�* 

Average reward per step\����9�       ��2	��
[̩�A�*

epsilon\���T-�.       ��W�	/[̩�A�* 

Average reward per step\����       ��2	�[̩�A�*

epsilon\����mx�.       ��W�	�k[̩�A�* 

Average reward per step\���S 4�       ��2	+l[̩�A�*

epsilon\����&��.       ��W�	�d[̩�A�* 

Average reward per step\���Jh�(       ��2	�e[̩�A�*

epsilon\���;|��.       ��W�	,c[̩�A�* 

Average reward per step\�����+a       ��2	�c[̩�A�*

epsilon\����:@�.       ��W�	[}[̩�A�* 

Average reward per step\�����j
       ��2	=~[̩�A�*

epsilon\���f]��.       ��W�	Y�[̩�A�* 

Average reward per step\����       ��2	L�[̩�A�*

epsilon\���;A!�0       ���_	��[̩�A*#
!
Average reward per episode��-��q�.       ��W�	S�[̩�A*!

total reward per episode  �J�2.       ��W�	ͯ[̩�A�* 

Average reward per step��-�(��       ��2	��[̩�A�*

epsilon��-�OZL�.       ��W�	y�[̩�A�* 

Average reward per step��-��Qߙ       ��2	%�[̩�A�*

epsilon��-��tC .       ��W�	��[̩�A�* 

Average reward per step��-��p��       ��2	x�[̩�A�*

epsilon��-�M�K.       ��W�	h�[̩�A�* 

Average reward per step��-�2��       ��2	��[̩�A�*

epsilon��-��gMU.       ��W�	�	"[̩�A�* 

Average reward per step��-���
�       ��2	$"[̩�A�*

epsilon��-��!�.       ��W�	kF$[̩�A�* 

Average reward per step��-�^^�       ��2	4G$[̩�A�*

epsilon��-�$Kϋ.       ��W�	�%[̩�A�* 

Average reward per step��-��`u�       ��2	�%[̩�A�*

epsilon��-�!-)�.       ��W�	�B'[̩�A�* 

Average reward per step��-��       ��2	�C'[̩�A�*

epsilon��-���۝.       ��W�	*�)[̩�A�* 

Average reward per step��-�N ��       ��2	�)[̩�A�*

epsilon��-���qe.       ��W�	T�+[̩�A�* 

Average reward per step��-���\>       ��2	;�+[̩�A�*

epsilon��-��G�9.       ��W�	.[̩�A�* 

Average reward per step��-�ńԇ       ��2	�.[̩�A�*

epsilon��-�zkO.       ��W�	=�1[̩�A�* 

Average reward per step��-�a�G       ��2	�1[̩�A�*

epsilon��-�T.       ��W�	�4[̩�A�* 

Average reward per step��-�Έ,       ��2	�4[̩�A�*

epsilon��-�&#�.       ��W�	��5[̩�A�* 

Average reward per step��-��t�       ��2	Z�5[̩�A�*

epsilon��-��S�W.       ��W�	P�7[̩�A�* 

Average reward per step��-���R�       ��2	��7[̩�A�*

epsilon��-��.�.       ��W�	��9[̩�A�* 

Average reward per step��-�=��       ��2	7�9[̩�A�*

epsilon��-��Kp�.       ��W�	� <[̩�A�* 

Average reward per step��-�gT>�       ��2	�<[̩�A�*

epsilon��-�"�D.       ��W�	6">[̩�A�* 

Average reward per step��-�V4E�       ��2	#>[̩�A�*

epsilon��-�q�zX.       ��W�	Z�?[̩�A�* 

Average reward per step��-���w*       ��2	�?[̩�A�*

epsilon��-����@.       ��W�	��A[̩�A�* 

Average reward per step��-��fB�       ��2	|�A[̩�A�*

epsilon��-�r	S.       ��W�	��C[̩�A�* 

Average reward per step��-�"�2       ��2	��C[̩�A�*

epsilon��-�'M9U.       ��W�	��F[̩�A�* 

Average reward per step��-�u       ��2	ؼF[̩�A�*

epsilon��-�p�.       ��W�	N
H[̩�A�* 

Average reward per step��-��_��       ��2	�
H[̩�A�*

epsilon��-���0\.       ��W�	J[̩�A�* 

Average reward per step��-�^�(�       ��2	�J[̩�A�*

epsilon��-��Kr.       ��W�	�L[̩�A�* 

Average reward per step��-�M�&�       ��2	�L[̩�A�*

epsilon��-��}�.       ��W�	mN[̩�A�* 

Average reward per step��-���       ��2	�mN[̩�A�*

epsilon��-�;�t�.       ��W�	W�O[̩�A�* 

Average reward per step��-�4��/       ��2	�O[̩�A�*

epsilon��-�T3�<.       ��W�	�Q[̩�A�* 

Average reward per step��-���oC       ��2	�Q[̩�A�*

epsilon��-��Ը.       ��W�	�S[̩�A�* 

Average reward per step��-��9&�       ��2	gS[̩�A�*

epsilon��-�06�h.       ��W�	|U[̩�A�* 

Average reward per step��-����       ��2	U[̩�A�*

epsilon��-�M�V�.       ��W�	�W[̩�A�* 

Average reward per step��-��/�       ��2	�W[̩�A�*

epsilon��-����.       ��W�	RY[̩�A�* 

Average reward per step��-��� �       ��2	,Y[̩�A�*

epsilon��-�A��.       ��W�	�A\[̩�A�* 

Average reward per step��-�;��       ��2	xB\[̩�A�*

epsilon��-��Bq�.       ��W�	߈_[̩�A�* 

Average reward per step��-����       ��2	��_[̩�A�*

epsilon��-�n��.       ��W�	Z�a[̩�A�* 

Average reward per step��-�0@f�       ��2	A�a[̩�A�*

epsilon��-����&.       ��W�	.sc[̩�A�* 

Average reward per step��-��G=�       ��2	tc[̩�A�*

epsilon��-���I}.       ��W�	�ve[̩�A�* 

Average reward per step��-���`�       ��2	�we[̩�A�*

epsilon��-��h�.       ��W�	��g[̩�A�* 

Average reward per step��-���P       ��2	h�g[̩�A�*

epsilon��-�[��.       ��W�	еi[̩�A�* 

Average reward per step��-�Ū/{       ��2	��i[̩�A�*

epsilon��-��� �.       ��W�	��k[̩�A�* 

Average reward per step��-���I       ��2	(�k[̩�A�*

epsilon��-��Q2�.       ��W�	��m[̩�A�* 

Average reward per step��-��7       ��2	\�m[̩�A�*

epsilon��-�� z�.       ��W�	`�o[̩�A�* 

Average reward per step��-��ϖ       ��2	��o[̩�A�*

epsilon��-�7��.       ��W�	H�q[̩�A�* 

Average reward per step��-��F��       ��2		�q[̩�A�*

epsilon��-��?|.       ��W�	�s[̩�A�* 

Average reward per step��-����       ��2	��s[̩�A�*

epsilon��-�;��2.       ��W�	i�u[̩�A�* 

Average reward per step��-�M/4�       ��2	%�u[̩�A�*

epsilon��-��@��.       ��W�	dWx[̩�A�* 

Average reward per step��-�.6       ��2	�Xx[̩�A�*

epsilon��-�IF�.       ��W�	��{[̩�A�* 

Average reward per step��-���~�       ��2	��{[̩�A�*

epsilon��-��a��.       ��W�	�}[̩�A�* 

Average reward per step��-�V��C       ��2	�}[̩�A�*

epsilon��-��p��.       ��W�	��[̩�A�* 

Average reward per step��-��/km       ��2	��[̩�A�*

epsilon��-�uϙ�.       ��W�	��[̩�A�* 

Average reward per step��-�9c�f       ��2	��[̩�A�*

epsilon��-�)'X�.       ��W�	��[̩�A�* 

Average reward per step��-�VI�       ��2	��[̩�A�*

epsilon��-�ESH\.       ��W�	��[̩�A�* 

Average reward per step��-�W �
       ��2	K�[̩�A�*

epsilon��-���A.       ��W�	��[̩�A�* 

Average reward per step��-�Z�|       ��2	���[̩�A�*

epsilon��-�\�T�.       ��W�	J�[̩�A�* 

Average reward per step��-���,       ��2	 	�[̩�A�*

epsilon��-��bd0       ���_	�-�[̩�A*#
!
Average reward per episodeh/���)�.       ��W�	'.�[̩�A*!

total reward per episode  ����^�.       ��W�	�`�[̩�A�* 

Average reward per steph/��_�       ��2	�a�[̩�A�*

epsilonh/῀���.       ��W�	Տ[̩�A�* 

Average reward per steph/����t       ��2	�Տ[̩�A�*

epsilonh/�#�Ś.       ��W�	r2�[̩�A�* 

Average reward per steph/Ή~��       ��2	a3�[̩�A�*

epsilonh/ῖ��2.       ��W�	�>�[̩�A�* 

Average reward per steph/Ῑe=�       ��2	hA�[̩�A�*

epsilonh/��]�R.       ��W�	ߕ[̩�A�* 

Average reward per steph/���4       ��2	�ߕ[̩�A�*

epsilonh/�D~�|.       ��W�	�#�[̩�A�* 

Average reward per steph/῅x7(       ��2	y$�[̩�A�*

epsilonh/�B~��.       ��W�	o*�[̩�A�* 

Average reward per steph/�5���       ��2	+�[̩�A�*

epsilonh/��^;.       ��W�	���[̩�A�* 

Average reward per steph/�E�Mr       ��2	f��[̩�A�*

epsilonh/��.       ��W�	)"�[̩�A�* 

Average reward per steph/�/r�F       ��2	�"�[̩�A�*

epsilonh/῅��.       ��W�	en�[̩�A�* 

Average reward per steph/�B�+       ��2	o�[̩�A�*

epsilonh/ῷu��.       ��W�	���[̩�A�* 

Average reward per steph/��b�       ��2	\��[̩�A�*

epsilonh/�fW�.       ��W�	���[̩�A�* 

Average reward per steph/�L.�8       ��2	���[̩�A�*

epsilonh/�VP��.       ��W�	���[̩�A�* 

Average reward per steph/�#��       ��2	���[̩�A�*

epsilonh/�v�i�.       ��W�	Q��[̩�A�* 

Average reward per steph/῀�6=       ��2	��[̩�A�*

epsilonh/�˝�.       ��W�	�I�[̩�A�* 

Average reward per steph/�6�g7       ��2	�J�[̩�A�*

epsilonh/ῆ�)e.       ��W�	���[̩�A�* 

Average reward per steph/�V�+       ��2	Y��[̩�A�*

epsilonh/��N��.       ��W�	��[̩�A�* 

Average reward per steph/����       ��2	��[̩�A�*

epsilonh/��3�.       ��W�	�[̩�A�* 

Average reward per steph/�_֓�       ��2	��[̩�A�*

epsilonh/Έ��.       ��W�	n�[̩�A�* 

Average reward per steph/�n       ��2	�n�[̩�A�*

epsilonh/ῇ�-�.       ��W�	ƨ�[̩�A�* 

Average reward per steph/΅�6S       ��2	���[̩�A�*

epsilonh/�f5&4.       ��W�	ᵳ[̩�A�* 

Average reward per steph/ῡ�7       ��2	���[̩�A�*

epsilonh/Έ;�z.       ��W�	�[̩�A�* 

Average reward per steph/�p��]       ��2	ȷ�[̩�A�*

epsilonh/�YIdX.       ��W�	Pȷ[̩�A�* 

Average reward per steph/�k�z�       ��2	2ɷ[̩�A�*

epsilonh/��QJ.       ��W�	��[̩�A�* 

Average reward per steph/ΐ��       ��2	���[̩�A�*

epsilonh/�ﷴ�.       ��W�	��[̩�A�* 

Average reward per steph/�.kR�       ��2	���[̩�A�*

epsilonh/�	<<.       ��W�	\�[̩�A�* 

Average reward per steph/�Tt�       ��2	��[̩�A�*

epsilonh/�����.       ��W�	��[̩�A�* 

Average reward per steph/�t_;�       ��2	��[̩�A�*

epsilonh/�O���.       ��W�	���[̩�A�* 

Average reward per steph/��q�       ��2	? �[̩�A�*

epsilonh/��S.       ��W�	���[̩�A�* 

Average reward per steph/�ю�       ��2	{��[̩�A�*

epsilonh/���.       ��W�	���[̩�A�* 

Average reward per steph/�f�V       ��2	I��[̩�A�*

epsilonh/�W�Og.       ��W�	f�[̩�A�* 

Average reward per steph/�@n�       ��2	�f�[̩�A�*

epsilonh/��m�.       ��W�	���[̩�A�* 

Average reward per steph/��7�       ��2	3��[̩�A�*

epsilonh/῱��`.       ��W�	���[̩�A�* 

Average reward per steph/��G�       ��2	`��[̩�A�*

epsilonh/��-�g.       ��W�	���[̩�A�* 

Average reward per steph/῍E�K       ��2	���[̩�A�*

epsilonh/῏��<.       ��W�	Eg�[̩�A�* 

Average reward per steph/��l$       ��2	@h�[̩�A�*

epsilonh/�%��.       ��W�	]k�[̩�A�* 

Average reward per steph/῎d�       ��2	�l�[̩�A�*

epsilonh/Ὼ���.       ��W�	� �[̩�A�* 

Average reward per steph/�%+       ��2	q�[̩�A�*

epsilonh/�Qۇ�.       ��W�	�K�[̩�A�* 

Average reward per steph/�/V�       ��2	bL�[̩�A�*

epsilonh/��qH5.       ��W�	(��[̩�A�* 

Average reward per steph/�C���       ��2	���[̩�A�*

epsilonh/��B�d.       ��W�	#��[̩�A�* 

Average reward per steph/��k��       ��2	'��[̩�A�*

epsilonh/ῖ��.       ��W�	���[̩�A�* 

Average reward per steph/῿�n�       ��2	���[̩�A�*

epsilonh/��N�V.       ��W�	O��[̩�A�* 

Average reward per steph/῔o�K       ��2	B��[̩�A�*

epsilonh/�[��I.       ��W�	o��[̩�A�* 

Average reward per steph/�(�d�       ��2	A��[̩�A�*

epsilonh/��8�.       ��W�	���[̩�A�* 

Average reward per steph/����       ��2	���[̩�A�*

epsilonh/�&��.       ��W�	��[̩�A�* 

Average reward per steph/�d�       ��2	��[̩�A�*

epsilonh/����Z.       ��W�	@k�[̩�A�* 

Average reward per steph/���j       ��2	+l�[̩�A�*

epsilonh/�q`�F.       ��W�	���[̩�A�* 

Average reward per steph/����       ��2	���[̩�A�*

epsilonh/�b]��.       ��W�	�[̩�A�* 

Average reward per steph/Ὸ��`       ��2	��[̩�A�*

epsilonh/��.       ��W�	+/�[̩�A�* 

Average reward per steph/�o       ��2	#0�[̩�A�*

epsilonh/�>&K>.       ��W�	Mj�[̩�A�* 

Average reward per steph/�"�F       ��2	Uk�[̩�A�*

epsilonh/�F��.       ��W�	���[̩�A�* 

Average reward per steph/�`�||       ��2	���[̩�A�*

epsilonh/��]�w.       ��W�	p"�[̩�A�* 

Average reward per steph/Ῠl�       ��2	}#�[̩�A�*

epsilonh/�;�G.       ��W�	1|�[̩�A�* 

Average reward per steph/ῄ�p�       ��2	�|�[̩�A�*

epsilonh/ῷȰ�.       ��W�	���[̩�A�* 

Average reward per steph/��B�       ��2	���[̩�A�*

epsilonh/��a�*.       ��W�	w1�[̩�A�* 

Average reward per steph/��r-�       ��2	U2�[̩�A�*

epsilonh/�{s^.       ��W�	�l \̩�A�* 

Average reward per steph/Ό�;g       ��2	�m \̩�A�*

epsilonh/��i.       ��W�	ge\̩�A�* 

Average reward per steph/�n��T       ��2	f\̩�A�*

epsilonh/�@3.       ��W�	Ȕ\̩�A�* 

Average reward per steph/ΰ�;3       ��2	y�\̩�A�*

epsilonh/�_z9.       ��W�	8�\̩�A�* 

Average reward per steph/���|       ��2	]�\̩�A�*

epsilonh/��ye .       ��W�	��\̩�A�* 

Average reward per steph/��[�       ��2	ɯ\̩�A�*

epsilonh/���#-0       ���_	_�\̩�A*#
!
Average reward per episode������`�.       ��W�	�\̩�A*!

total reward per episode  ��:c/F.       ��W�	��\̩�A�* 

Average reward per step�������       ��2	��\̩�A�*

epsilon�����r�.       ��W�	oe\̩�A�* 

Average reward per step�����<ŋ       ��2	<f\̩�A�*

epsilon�����lVC.       ��W�	�b\̩�A�* 

Average reward per step������D)       ��2	�c\̩�A�*

epsilon�����X�.       ��W�	�1\̩�A�* 

Average reward per step�����A�       ��2	�2\̩�A�*

epsilon����#b(�.       ��W�	B�\̩�A�* 

Average reward per step�������       ��2	г\̩�A�*

epsilon����}SK.       ��W�	 �\̩�A�* 

Average reward per step����u'�       ��2	֪\̩�A�*

epsilon������t8.       ��W�	��\̩�A�* 

Average reward per step�������       ��2	��\̩�A�*

epsilon����Ȅ��.       ��W�	��\̩�A�* 

Average reward per step����N7
�       ��2	��\̩�A�*

epsilon����cG4.       ��W�	��\̩�A�* 

Average reward per step����"cs       ��2	�\̩�A�*

epsilon����jĉ.       ��W�	C;\̩�A�* 

Average reward per step����a/|2       ��2	<\̩�A�*

epsilon�����gd�.       ��W�	�6 \̩�A�* 

Average reward per step����)9z�       ��2	�7 \̩�A�*

epsilon����T��.       ��W�	��!\̩�A�* 

Average reward per step�����Gd       ��2	A�!\̩�A�*

epsilon�����K.       ��W�	�#\̩�A�* 

Average reward per step�����J;�       ��2	�#\̩�A�*

epsilon����1���.       ��W�	h>%\̩�A�* 

Average reward per step�����?       ��2	)?%\̩�A�*

epsilon����;�.       ��W�	.<'\̩�A�* 

Average reward per step����W�*�       ��2	�<'\̩�A�*

epsilon����P]}�.       ��W�	�K)\̩�A�* 

Average reward per step����W�\*       ��2	UL)\̩�A�*

epsilon������.       ��W�	�W+\̩�A�* 

Average reward per step�����i��       ��2	�X+\̩�A�*

epsilon����e�(s.       ��W�	�-\̩�A�* 

Average reward per step����]VT�       ��2	Ք-\̩�A�*

epsilon������9�.       ��W�	7�/\̩�A�* 

Average reward per step�����4��       ��2	Ɏ/\̩�A�*

epsilon������q	.       ��W�	k'1\̩�A�* 

Average reward per step�������a       ��2	�'1\̩�A�*

epsilon������y�0       ���_	�A1\̩�A*#
!
Average reward per episode33���Bt.       ��W�	-B1\̩�A*!

total reward per episode  �9Υ^.       ��W�	��5\̩�A�* 

Average reward per step33���4�       ��2	g�5\̩�A�*

epsilon33��&��o.       ��W�	��7\̩�A�* 

Average reward per step33�����v       ��2	D�7\̩�A�*

epsilon33���4).       ��W�	U�9\̩�A�* 

Average reward per step33�����p       ��2	/�9\̩�A�*

epsilon33����|.       ��W�	��;\̩�A�* 

Average reward per step33�����       ��2	�;\̩�A�*

epsilon33��۾��.       ��W�	�q=\̩�A�* 

Average reward per step33�����       ��2	�s=\̩�A�*

epsilon33��([��.       ��W�	�p?\̩�A�* 

Average reward per step33���w�       ��2	�q?\̩�A�*

epsilon33��,�;_.       ��W�	�oA\̩�A�* 

Average reward per step33����I�       ��2	�pA\̩�A�*

epsilon33���R6.       ��W�	4D\̩�A�* 

Average reward per step33���Gf       ��2	�4D\̩�A�*

epsilon33��:R\.       ��W�	0�E\̩�A�* 

Average reward per step33���c �       ��2	��E\̩�A�*

epsilon33����B.       ��W�	H\̩�A�* 

Average reward per step33��A��       ��2	�H\̩�A�*

epsilon33��;	Y�.       ��W�	�J\̩�A�* 

Average reward per step33��kff       ��2	�J\̩�A�*

epsilon33����[.       ��W�	�HL\̩�A�* 

Average reward per step33���n"�       ��2	�IL\̩�A�*

epsilon33���\�E.       ��W�	��M\̩�A�* 

Average reward per step33������       ��2	[�M\̩�A�*

epsilon33��w�b..       ��W�	��O\̩�A�* 

Average reward per step33�����w       ��2	��O\̩�A�*

epsilon33��XrO�.       ��W�	�Q\̩�A�* 

Average reward per step33��qȨ|       ��2	��Q\̩�A�*

epsilon33���%�.       ��W�	Z�T\̩�A�* 

Average reward per step33�����7       ��2	�T\̩�A�*

epsilon33��*"H9.       ��W�	^X\̩�A�* 

Average reward per step33���!�       ��2	X\̩�A�*

epsilon33��0"�.       ��W�	:Z\̩�A�* 

Average reward per step33��8��       ��2	�Z\̩�A�*

epsilon33�����.       ��W�	i;\\̩�A�* 

Average reward per step33��(�9W       ��2	<\\̩�A�*

epsilon33��x��.       ��W�	�6^\̩�A�* 

Average reward per step33���7�7       ��2	�7^\̩�A�*

epsilon33��ZEr�.       ��W�	�N`\̩�A�* 

Average reward per step33��-,J�       ��2	rO`\̩�A�*

epsilon33��QF.       ��W�	
Jb\̩�A�* 

Average reward per step33��P���       ��2	�Jb\̩�A�*

epsilon33���87.       ��W�	Pd\̩�A�* 

Average reward per step33��~Z*       ��2	�Pd\̩�A�*

epsilon33��T��.       ��W�	af\̩�A�* 

Average reward per step33���9^       ��2	�af\̩�A�*

epsilon33��9H�.       ��W�	p^h\̩�A�* 

Average reward per step33���ʍ       ��2	__h\̩�A�*

epsilon33��k�.       ��W�	f�j\̩�A�* 

Average reward per step33���]��       ��2	#�j\̩�A�*

epsilon33��{��7.       ��W�	Ֆl\̩�A�* 

Average reward per step33��
B0/       ��2	Зl\̩�A�*

epsilon33��Q�a.       ��W�	e�n\̩�A�* 

Average reward per step33�����D       ��2	"�n\̩�A�*

epsilon33��~ɲ�.       ��W�	��p\̩�A�* 

Average reward per step33��1�       ��2	B�p\̩�A�*

epsilon33���ʪ�.       ��W�	�r\̩�A�* 

Average reward per step33���r��       ��2	��r\̩�A�*

epsilon33���d)$0       ���_	Ls\̩�A	*#
!
Average reward per episode  ��'�e.       ��W�	�s\̩�A	*!

total reward per episode  �^a�>.       ��W�	k�v\̩�A�* 

Average reward per step  �����       ��2	Z�v\̩�A�*

epsilon  ����.       ��W�	�y\̩�A�* 

Average reward per step  ����M       ��2	�y\̩�A�*

epsilon  ��kXj.       ��W�	�ez\̩�A�* 

Average reward per step  ���t�w       ��2	�fz\̩�A�*

epsilon  ��1��.       ��W�	��{\̩�A�* 

Average reward per step  �����       ��2	��{\̩�A�*

epsilon  ���'.       ��W�	{�}\̩�A�* 

Average reward per step  ��%к�       ��2	'�}\̩�A�*

epsilon  ���H�2.       ��W�	(�\̩�A�* 

Average reward per step  ��a��       ��2	�\̩�A�*

epsilon  ��}�je.       ��W�	��\̩�A�* 

Average reward per step  ����J-       ��2	��\̩�A�*

epsilon  ����~�.       ��W�	��\̩�A�* 

Average reward per step  ���"B       ��2	%�\̩�A�*

epsilon  ���Fv.       ��W�	6v�\̩�A�* 

Average reward per step  ���\       ��2	Ww�\̩�A�*

epsilon  ��ZW{D.       ��W�	�u�\̩�A�* 

Average reward per step  ����p       ��2	�v�\̩�A�*

epsilon  ���H�.       ��W�	c��\̩�A�* 

Average reward per step  ���n��       ��2	R��\̩�A�*

epsilon  ��,�i�.       ��W�	���\̩�A�* 

Average reward per step  ��T���       ��2	$��\̩�A�*

epsilon  ����O=0       ���_	r��\̩�A
*#
!
Average reward per episode  \����.       ��W�	���\̩�A
*!

total reward per episode  %ü�q.       ��W�	^-�\̩�A�* 

Average reward per step  \��[�S       ��2	,.�\̩�A�*

epsilon  \�=��k.       ��W�	���\̩�A�* 

Average reward per step  \��|&>       ��2	}��\̩�A�*

epsilon  \�[9.       ��W�	N��\̩�A�* 

Average reward per step  \�c��       ��2	=��\̩�A�*

epsilon  \�[��.       ��W�	���\̩�A�* 

Average reward per step  \�MI��       ��2	s��\̩�A�*

epsilon  \��?�7.       ��W�	[�\̩�A�* 

Average reward per step  \����%       ��2	��\̩�A�*

epsilon  \����.       ��W�	h?�\̩�A�* 

Average reward per step  \��c       ��2	B@�\̩�A�*

epsilon  \��(Hp.       ��W�	K��\̩�A�* 

Average reward per step  \�����       ��2	���\̩�A�*

epsilon  \�KW�.       ��W�	��\̩�A�* 

Average reward per step  \���m       ��2	r�\̩�A�*

epsilon  \��$��.       ��W�	5}�\̩�A�* 

Average reward per step  \��*�s       ��2	~�\̩�A�*

epsilon  \��5��.       ��W�		��\̩�A�* 

Average reward per step  \�3��|       ��2	ߌ�\̩�A�*

epsilon  \�[�^�.       ��W�	�;�\̩�A�* 

Average reward per step  \��S�       ��2	m<�\̩�A�*

epsilon  \�~dA�.       ��W�	*U�\̩�A�* 

Average reward per step  \�o�*       ��2	V�\̩�A�*

epsilon  \��l��.       ��W�	=��\̩�A�* 

Average reward per step  \����(       ��2	,��\̩�A�*

epsilon  \��	.       ��W�	�Ǫ\̩�A�* 

Average reward per step  \���8       ��2	�Ȫ\̩�A�*

epsilon  \���.       ��W�	̬\̩�A�* 

Average reward per step  \��e       ��2	�̬\̩�A�*

epsilon  \���m�.       ��W�	D��\̩�A�* 

Average reward per step  \��H9�       ��2	'��\̩�A�*

epsilon  \�b��.       ��W�	%̰\̩�A�* 

Average reward per step  \��n�F       ��2	Ͱ\̩�A�*

epsilon  \���w�.       ��W�	�q�\̩�A�* 

Average reward per step  \�^��w       ��2	Tr�\̩�A�*

epsilon  \����0       ���_	c��\̩�A*#
!
Average reward per episode����u��.       ��W�	��\̩�A*!

total reward per episode  #��v��.       ��W�	�%�\̩�A�* 

Average reward per step���B$�       ��2	�&�\̩�A�*

epsilon���Z82:.       ��W�	d#�\̩�A�* 

Average reward per step������       ��2	F$�\̩�A�*

epsilon������".       ��W�	��\̩�A�* 

Average reward per step���-_t�       ��2	��\̩�A�*

epsilon���L{�.       ��W�	��\̩�A�* 

Average reward per step���S`2       ��2	n��\̩�A�*

epsilon���2��.       ��W�	1#�\̩�A�* 

Average reward per step����ʑs       ��2	�#�\̩�A�*

epsilon���b&�.       ��W�	��\̩�A�* 

Average reward per step������t       ��2	^�\̩�A�*

epsilon�����HS.       ��W�	�E�\̩�A�* 

Average reward per step�������       ��2	_F�\̩�A�*

epsilon���S8j.       ��W�	��\̩�A�* 

Average reward per step�����U|       ��2	;��\̩�A�*

epsilon����K��.       ��W�	#��\̩�A�* 

Average reward per step���-(F�       ��2	��\̩�A�*

epsilon���w/.       ��W�	�M�\̩�A�* 

Average reward per step���A6�       ��2	�N�\̩�A�*

epsilon���OS~.       ��W�	bN�\̩�A�* 

Average reward per step���˳�c       ��2	<O�\̩�A�*

epsilon���H��.       ��W�	�x�\̩�A�* 

Average reward per step����w�       ��2	�y�\̩�A�*

epsilon����i.       ��W�	���\̩�A�* 

Average reward per step���A��       ��2	���\̩�A�*

epsilon�������.       ��W�	��\̩�A�* 

Average reward per step����.n       ��2	Ǻ�\̩�A�*

epsilon������0       ���_	���\̩�A*#
!
Average reward per episode�$1��B��.       ��W�	D��\̩�A*!

total reward per episode  �u)1.       ��W�	���\̩�A�* 

Average reward per step�$1��7�8       ��2	,��\̩�A�*

epsilon�$1�K\�".       ��W�	 ��\̩�A�* 

Average reward per step�$1�
�Eu       ��2	Ҍ�\̩�A�*

epsilon�$1��m�.       ��W�	��\̩�A�* 

Average reward per step�$1�r]�       ��2	���\̩�A�*

epsilon�$1���H.       ��W�	Ō�\̩�A�* 

Average reward per step�$1����       ��2	���\̩�A�*

epsilon�$1��.       ��W�	��\̩�A�* 

Average reward per step�$1����a       ��2	���\̩�A�*

epsilon�$1����.       ��W�	z��\̩�A�* 

Average reward per step�$1��Uec       ��2	?��\̩�A�*

epsilon�$1����.       ��W�	J{�\̩�A�* 

Average reward per step�$1��Z�z       ��2	B|�\̩�A�*

epsilon�$1���Q.       ��W�	���\̩�A�* 

Average reward per step�$1��!�T       ��2	���\̩�A�*

epsilon�$1�EmG�.       ��W�	�>�\̩�A�* 

Average reward per step�$1���2r       ��2	�?�\̩�A�*

epsilon�$1�iS.       ��W�	l��\̩�A�* 

Average reward per step�$1�!a�	       ��2	���\̩�A�*

epsilon�$1�{�F.       ��W�	���\̩�A�* 

Average reward per step�$1���p       ��2	ػ�\̩�A�*

epsilon�$1�$�i�.       ��W�	n��\̩�A�* 

Average reward per step�$1���-       ��2	��\̩�A�*

epsilon�$1��l.       ��W�	��\̩�A�* 

Average reward per step�$1� ��       ��2	Ͻ�\̩�A�*

epsilon�$1����/.       ��W�	��\̩�A�* 

Average reward per step�$1�H��s       ��2	�\̩�A�*

epsilon�$1����.       ��W�	�1�\̩�A�* 

Average reward per step�$1��]M"       ��2	�2�\̩�A�*

epsilon�$1�I�k.       ��W�	SY�\̩�A�* 

Average reward per step�$1�X�       ��2	�Y�\̩�A�*

epsilon�$1��+�.       ��W�	��\̩�A�* 

Average reward per step�$1��S�       ��2	S�\̩�A�*

epsilon�$1����.       ��W�	Lq�\̩�A�* 

Average reward per step�$1�UGP^       ��2	�s�\̩�A�*

epsilon�$1����8.       ��W�	�\̩�A�* 

Average reward per step�$1�d�0�       ��2	���\̩�A�*

epsilon�$1��c��.       ��W�	���\̩�A�* 

Average reward per step�$1�bb+X       ��2	���\̩�A�*

epsilon�$1��VI.       ��W�	���\̩�A�* 

Average reward per step�$1�y�f�       ��2	t��\̩�A�*

epsilon�$1��4:.       ��W�	 F ]̩�A�* 

Average reward per step�$1�7�*       ��2	,G ]̩�A�*

epsilon�$1�L-�E.       ��W�	��]̩�A�* 

Average reward per step�$1��rsd       ��2	y�]̩�A�*

epsilon�$1�]�j.       ��W�	t�]̩�A�* 

Average reward per step�$1�"�*7       ��2	E�]̩�A�*

epsilon�$1�q'Y�.       ��W�	��]̩�A�* 

Average reward per step�$1���O       ��2	��]̩�A�*

epsilon�$1�<W�.       ��W�	+�	]̩�A�* 

Average reward per step�$1�l�K�       ��2	�	]̩�A�*

epsilon�$1�nu��.       ��W�	��]̩�A�* 

Average reward per step�$1�^�Yg       ��2	��]̩�A�*

epsilon�$1���$�.       ��W�	]̩�A�* 

Average reward per step�$1�x�       ��2	�]̩�A�*

epsilon�$1����.       ��W�	{]̩�A�* 

Average reward per step�$1�7,?       ��2	]̩�A�*

epsilon�$1��!q.       ��W�	E]̩�A�* 

Average reward per step�$1�\82�       ��2	,]̩�A�*

epsilon�$1�#�L.       ��W�	�G]̩�A�* 

Average reward per step�$1��fZo       ��2	sH]̩�A�*

epsilon�$1����.       ��W�	XY]̩�A�* 

Average reward per step�$1�U!;�       ��2	�Y]̩�A�*

epsilon�$1��d�c.       ��W�	Ps]̩�A�* 

Average reward per step�$1�(y�       ��2	�s]̩�A�*

epsilon�$1�U��.       ��W�	>�]̩�A�* 

Average reward per step�$1���V       ��2	��]̩�A�*

epsilon�$1�����.       ��W�	^K]̩�A�* 

Average reward per step�$1�J.�;       ��2	YL]̩�A�*

epsilon�$1�J�.       ��W�	�:]̩�A�* 

Average reward per step�$1�`:�       ��2	�;]̩�A�*

epsilon�$1����.       ��W�	 T]̩�A�* 

Average reward per step�$1���T       ��2	�T]̩�A�*

epsilon�$1��p�.       ��W�	�b!]̩�A�* 

Average reward per step�$1�{y*       ��2	Vc!]̩�A�*

epsilon�$1�(���.       ��W�	�u#]̩�A�* 

Average reward per step�$1�?��O       ��2	yv#]̩�A�*

epsilon�$1����g0       ���_	_�#]̩�A*#
!
Average reward per episodeH�4���& .       ��W�	ݖ#]̩�A*!

total reward per episode  ���� .       ��W�	��']̩�A�* 

Average reward per stepH�4�O�VF       ��2	-�']̩�A�*

epsilonH�4����.       ��W�	ӣ)]̩�A�* 

Average reward per stepH�4��h��       ��2	��)]̩�A�*

epsilonH�4�����.       ��W�	1�+]̩�A�* 

Average reward per stepH�4��}F#       ��2	ͱ+]̩�A�*

epsilonH�4���q.       ��W�	�-]̩�A�* 

Average reward per stepH�4�q���       ��2	��-]̩�A�*

epsilonH�4�
p4.       ��W�	��/]̩�A�* 

Average reward per stepH�4���S       ��2	f�/]̩�A�*

epsilonH�4�M�Q.       ��W�	��1]̩�A�* 

Average reward per stepH�4����       ��2	��1]̩�A�*

epsilonH�4�"�]�.       ��W�	�3]̩�A�* 

Average reward per stepH�4��o       ��2	��3]̩�A�*

epsilonH�4��
ٱ.       ��W�	u5]̩�A�* 

Average reward per stepH�4����       ��2	v5]̩�A�*

epsilonH�4���*�.       ��W�	m7]̩�A�* 

Average reward per stepH�4�ޛt�       ��2	�m7]̩�A�*

epsilonH�4���.       ��W�	�8]̩�A�* 

Average reward per stepH�4��W��       ��2	�8]̩�A�*

epsilonH�4�^zm�.       ��W�	$
;]̩�A�* 

Average reward per stepH�4��U4       ��2	�
;]̩�A�*

epsilonH�4���.       ��W�	�|<]̩�A�* 

Average reward per stepH�4����[       ��2	V}<]̩�A�*

epsilonH�4���;p.       ��W�	I�=]̩�A�* 

Average reward per stepH�4�-�r       ��2	��=]̩�A�*

epsilonH�4�!.       ��W�	2@]̩�A�* 

Average reward per stepH�4�Cx�       ��2	@]̩�A�*

epsilonH�4����r.       ��W�	��A]̩�A�* 

Average reward per stepH�4����S       ��2	K�A]̩�A�*

epsilonH�4���.       ��W�	��B]̩�A�* 

Average reward per stepH�4�����       ��2	��B]̩�A�*

epsilonH�4�����.       ��W�	V�D]̩�A�* 

Average reward per stepH�4�=N�       ��2	=�D]̩�A�*

epsilonH�4�G��0       ���_	�E]̩�A*#
!
Average reward per episodeZZ��<�).       ��W�	�E]̩�A*!

total reward per episode  $éW�.       ��W�	J�H]̩�A�* 

Average reward per stepZZ�	��       ��2	�H]̩�A�*

epsilonZZ�=ٶ�.       ��W�	-�J]̩�A�* 

Average reward per stepZZ�,[�       ��2	�J]̩�A�*

epsilonZZ����.       ��W�	��M]̩�A�* 

Average reward per stepZZ�.�w5       ��2	��M]̩�A�*

epsilonZZ��\�S.       ��W�	nO]̩�A�* 

Average reward per stepZZ��8�4       ��2	�O]̩�A�*

epsilonZZ�NS	.       ��W�	Q]̩�A�* 

Average reward per stepZZ����       ��2	�Q]̩�A�*

epsilonZZ��N�.       ��W�	"9S]̩�A�* 

Average reward per stepZZ��ŋ3       ��2	�9S]̩�A�*

epsilonZZ�TMhx.       ��W�	�1U]̩�A�* 

Average reward per stepZZ��j�       ��2	�2U]̩�A�*

epsilonZZ���]^.       ��W�	�8W]̩�A�* 

Average reward per stepZZ���6�       ��2	�9W]̩�A�*

epsilonZZ��9i�.       ��W�	oEY]̩�A�* 

Average reward per stepZZ���3       ��2	AFY]̩�A�*

epsilonZZ����'.       ��W�	�F[]̩�A�* 

Average reward per stepZZ���}       ��2	0G[]̩�A�*

epsilonZZ���T�.       ��W�	EF]]̩�A�* 

Average reward per stepZZ�7=z       ��2	�F]]̩�A�*

epsilonZZ���u.       ��W�	�b_]̩�A�* 

Average reward per stepZZ��>�       ��2	�c_]̩�A�*

epsilonZZ��.H�.       ��W�	�ha]̩�A�* 

Average reward per stepZZ��,^
       ��2	�ia]̩�A�*

epsilonZZ��>#�.       ��W�	��c]̩�A�* 

Average reward per stepZZ��1�       ��2	��c]̩�A�*

epsilonZZ����!.       ��W�	@�e]̩�A�* 

Average reward per stepZZ�t3^       ��2	�e]̩�A�*

epsilonZZ�|�9�.       ��W�	d�g]̩�A�* 

Average reward per stepZZ�c���       ��2	��g]̩�A�*

epsilonZZ��\.       ��W�	��i]̩�A�* 

Average reward per stepZZ�]/�       ��2	�i]̩�A�*

epsilonZZ�ǳ�.       ��W�	�k]̩�A�* 

Average reward per stepZZ�X�Rq       ��2	S�k]̩�A�*

epsilonZZ� I;..       ��W�	�n]̩�A�* 

Average reward per stepZZ��g��       ��2	Hn]̩�A�*

epsilonZZ��pJ.       ��W�	�p]̩�A�* 

Average reward per stepZZ�t�@K       ��2	�p]̩�A�*

epsilonZZ�;J�.       ��W�	�r]̩�A�* 

Average reward per stepZZ�m�       ��2	�r]̩�A�*

epsilonZZ��:�W.       ��W�	�Ot]̩�A�* 

Average reward per stepZZ��S�       ��2	�Pt]̩�A�*

epsilonZZ����.       ��W�	�u]̩�A�* 

Average reward per stepZZ�.���       ��2	Ûu]̩�A�*

epsilonZZ����.       ��W�	؛w]̩�A�* 

Average reward per stepZZ��x��       ��2	s�w]̩�A�*

epsilonZZ��C�.       ��W�	ͯy]̩�A�* 

Average reward per stepZZ�Xq�       ��2	��y]̩�A�*

epsilonZZ�NT�Q.       ��W�	a�{]̩�A�* 

Average reward per stepZZ�¾��       ��2	?�{]̩�A�*

epsilonZZ�/]�.       ��W�	@�}]̩�A�* 

Average reward per stepZZ�YT�9       ��2	��}]̩�A�*

epsilonZZ���n�.       ��W�	d=�]̩�A�* 

Average reward per stepZZ�=(�#       ��2	2>�]̩�A�*

epsilonZZ��*E�.       ��W�	�C�]̩�A�* 

Average reward per stepZZ�וc$       ��2	5E�]̩�A�*

epsilonZZ�;,ZV.       ��W�	�n�]̩�A�* 

Average reward per stepZZ�H���       ��2	�o�]̩�A�*

epsilonZZ��w.       ��W�	uǅ]̩�A�* 

Average reward per stepZZ��{�       ��2	Pȅ]̩�A�*

epsilonZZ�'v�_.       ��W�	V�]̩�A�* 

Average reward per stepZZ�/}�       ��2	'�]̩�A�*

epsilonZZ�wc��.       ��W�	>&�]̩�A�* 

Average reward per stepZZ�0ߩ�       ��2	�&�]̩�A�*

epsilonZZ��!j.       ��W�	D�]̩�A�* 

Average reward per stepZZ�_��       ��2	�D�]̩�A�*

epsilonZZ�iՖ%.       ��W�	��]̩�A�* 

Average reward per stepZZ�����       ��2	I�]̩�A�*

epsilonZZ�pS�".       ��W�	�7�]̩�A�* 

Average reward per stepZZ� ��e       ��2	m8�]̩�A�*

epsilonZZ�X	��.       ��W�	�c�]̩�A�* 

Average reward per stepZZ�\Փ       ��2	�d�]̩�A�*

epsilonZZ���V.       ��W�	ob�]̩�A�* 

Average reward per stepZZ��r�       ��2	c�]̩�A�*

epsilonZZ���w�.       ��W�	�K�]̩�A�* 

Average reward per stepZZ���       ��2	'L�]̩�A�*

epsilonZZ�Z(��.       ��W�	KW�]̩�A�* 

Average reward per stepZZ�W.%       ��2	%X�]̩�A�*

epsilonZZ�ZU�.       ��W�	���]̩�A�* 

Average reward per stepZZ��NA       ��2	o��]̩�A�*

epsilonZZ��#�.       ��W�	؝�]̩�A�* 

Average reward per stepZZ����"       ��2	���]̩�A�*

epsilonZZ���.       ��W�	���]̩�A�* 

Average reward per stepZZ�9ӓ       ��2	���]̩�A�*

epsilonZZ��mi�.       ��W�	7S�]̩�A�* 

Average reward per stepZZ��Lz2       ��2	T�]̩�A�*

epsilonZZ�	�<.       ��W�	��]̩�A�* 

Average reward per stepZZ��ِ�       ��2	�]̩�A�*

epsilonZZ�Xf�.       ��W�	D��]̩�A�* 

Average reward per stepZZ�-]4       ��2	��]̩�A�*

epsilonZZ��g�.       ��W�	Z��]̩�A�* 

Average reward per stepZZ�@a_P       ��2	s��]̩�A�*

epsilonZZ�I{�.       ��W�	�Ч]̩�A�* 

Average reward per stepZZ�x6�       ��2	�ѧ]̩�A�*

epsilonZZ�q-�W.       ��W�	T�]̩�A�* 

Average reward per stepZZ�$L{       ��2	�]̩�A�*

epsilonZZ��ϛ.       ��W�	��]̩�A�* 

Average reward per stepZZ���       ��2	��]̩�A�*

epsilonZZ��p��.       ��W�	Q��]̩�A�* 

Average reward per stepZZ�\V�)       ��2	+��]̩�A�*

epsilonZZ�?���.       ��W�	j��]̩�A�* 

Average reward per stepZZ��I1       ��2	H��]̩�A�*

epsilonZZ���~.       ��W�	4��]̩�A�* 

Average reward per stepZZ�w�[7       ��2	
��]̩�A�*

epsilonZZ��2�.       ��W�	���]̩�A�* 

Average reward per stepZZ��&5:       ��2	� �]̩�A�*

epsilonZZ���s{.       ��W�	��]̩�A�* 

Average reward per stepZZ�_+�       ��2	��]̩�A�*

epsilonZZ��)`�.       ��W�	p#�]̩�A�* 

Average reward per stepZZ�|��F       ��2	5$�]̩�A�*

epsilonZZ�I���.       ��W�	E)�]̩�A�* 

Average reward per stepZZ�D�N�       ��2	�)�]̩�A�*

epsilonZZ��7�.       ��W�	�4�]̩�A�* 

Average reward per stepZZ��ř�       ��2	�5�]̩�A�*

epsilonZZ����e.       ��W�	P�]̩�A�* 

Average reward per stepZZ��y0�       ��2	�P�]̩�A�*

epsilonZZ�Rf9�.       ��W�	�`�]̩�A�* 

Average reward per stepZZ����f       ��2	xa�]̩�A�*

epsilonZZ��W�U.       ��W�	�T�]̩�A�* 

Average reward per stepZZ�,��       ��2	?U�]̩�A�*

epsilonZZ���.       ��W�	x_�]̩�A�* 

Average reward per stepZZ��up�       ��2	_`�]̩�A�*

epsilonZZ�ܥ��.       ��W�	��]̩�A�* 

Average reward per stepZZ�d�ء       ��2	Ŭ�]̩�A�*

epsilonZZ��ZXl.       ��W�	b��]̩�A�* 

Average reward per stepZZ�f��o       ��2	���]̩�A�*

epsilonZZ�M�.       ��W�	��]̩�A�* 

Average reward per stepZZ�*�b�       ��2	X�]̩�A�*

epsilonZZ�����0       ���_	T:�]̩�A*#
!
Average reward per episode�Nl�1�H�.       ��W�	�:�]̩�A*!

total reward per episode  p�CM{O.       ��W�	Л�]̩�A�* 

Average reward per step�Nl��a       ��2	���]̩�A�*

epsilon�Nl��腔.       ��W�		��]̩�A�* 

Average reward per step�Nl�P�       ��2	���]̩�A�*

epsilon�Nl�؏��.       ��W�	��]̩�A�* 

Average reward per step�Nl��m^@       ��2	���]̩�A�*

epsilon�Nl�4�Z�.       ��W�	���]̩�A�* 

Average reward per step�Nl��NfA       ��2	\��]̩�A�*

epsilon�Nl��//=.       ��W�	y��]̩�A�* 

Average reward per step�Nl�Xq[A       ��2	��]̩�A�*

epsilon�Nl���QM.       ��W�	��]̩�A�* 

Average reward per step�Nl�h��       ��2	���]̩�A�*

epsilon�Nl��/�.       ��W�	��]̩�A�* 

Average reward per step�Nl���       ��2	��]̩�A�*

epsilon�Nl�W�e.       ��W�	ʨ�]̩�A�* 

Average reward per step�Nl�?��       ��2	q��]̩�A�*

epsilon�Nl�?��.       ��W�	���]̩�A�* 

Average reward per step�Nl�~	{I       ��2	���]̩�A�*

epsilon�Nl��_@|.       ��W�	<�]̩�A�* 

Average reward per step�Nl��&       ��2	�]̩�A�*

epsilon�Nl�)BA
.       ��W�	{3�]̩�A�* 

Average reward per step�Nl�]0��       ��2	U4�]̩�A�*

epsilon�Nl�E���.       ��W�	f��]̩�A�* 

Average reward per step�Nl�|$C       ��2	��]̩�A�*

epsilon�Nl�q{ �.       ��W�	��]̩�A�* 

Average reward per step�Nl��jk       ��2	���]̩�A�*

epsilon�Nl�Q�*�.       ��W�	���]̩�A�* 

Average reward per step�Nl�a#x�       ��2	���]̩�A�*

epsilon�Nl���O.       ��W�	t��]̩�A�* 

Average reward per step�Nl�[       ��2	>��]̩�A�*

epsilon�Nl��f��.       ��W�	� �]̩�A�* 

Average reward per step�Nl���       ��2	��]̩�A�*

epsilon�Nl�뺢�.       ��W�	��]̩�A�* 

Average reward per step�Nl�Ed�       ��2	��]̩�A�*

epsilon�Nl�T	�b.       ��W�	���]̩�A�* 

Average reward per step�Nl�ο       ��2	� �]̩�A�*

epsilon�Nl����w.       ��W�	
�]̩�A�* 

Average reward per step�Nl���c       ��2	��]̩�A�*

epsilon�Nl��o��.       ��W�	�>�]̩�A�* 

Average reward per step�Nl���       ��2	l?�]̩�A�*

epsilon�Nl� ���.       ��W�	�=�]̩�A�* 

Average reward per step�Nl�.�D       ��2	u>�]̩�A�*

epsilon�Nl����.       ��W�	]�]̩�A�* 

Average reward per step�Nl�X�&       ��2	�]�]̩�A�*

epsilon�Nl�1.��.       ��W�	#��]̩�A�* 

Average reward per step�Nl��_�       ��2	���]̩�A�*

epsilon�Nl��`��.       ��W�	���]̩�A�* 

Average reward per step�Nl�x�s�       ��2	g��]̩�A�*

epsilon�Nl�bےi.       ��W�	��]̩�A�* 

Average reward per step�Nl���_       ��2	v�]̩�A�*

epsilon�Nl�u��.       ��W�	^̩�A�* 

Average reward per step�Nl��3��       ��2	�^̩�A�*

epsilon�Nl�yGԢ.       ��W�	�0^̩�A�* 

Average reward per step�Nl�L�B�       ��2	f1^̩�A�*

epsilon�Nl�3|�.       ��W�	}<^̩�A�* 

Average reward per step�Nl���k�       ��2	`=^̩�A�*

epsilon�Nl�̩Z�.       ��W�	�m^̩�A�* 

Average reward per step�Nl��(�       ��2	zn^̩�A�*

epsilon�Nl�&��D.       ��W�	'l	^̩�A�* 

Average reward per step�Nl�?a[�       ��2	�l	^̩�A�*

epsilon�Nl��K@�.       ��W�	U�^̩�A�* 

Average reward per step�Nl�τ       ��2	'�^̩�A�*

epsilon�Nl��~�..       ��W�	�~^̩�A�* 

Average reward per step�Nl�(�;\       ��2	�^̩�A�*

epsilon�Nl�7m,.       ��W�	�?^̩�A�* 

Average reward per step�Nl�"�Χ       ��2	�@^̩�A�*

epsilon�Nl�K�6�.       ��W�	wM^̩�A�* 

Average reward per step�Nl��L�9       ��2	MN^̩�A�*

epsilon�Nl���vD.       ��W�	%�^̩�A�* 

Average reward per step�Nl�}�I       ��2	��^̩�A�*

epsilon�Nl�0�.       ��W�	��^̩�A�* 

Average reward per step�Nl���IC       ��2	{�^̩�A�*

epsilon�Nl�-1�e.       ��W�	��^̩�A�* 

Average reward per step�Nl��(�       ��2	)�^̩�A�*

epsilon�Nl��	��.       ��W�	s�^̩�A�* 

Average reward per step�Nl�SO �       ��2	�^̩�A�*

epsilon�Nl��rG.       ��W�	��^̩�A�* 

Average reward per step�Nl��s-       ��2	U�^̩�A�*

epsilon�Nl�-f��.       ��W�	S^̩�A�* 

Average reward per step�Nl��� [       ��2	1^̩�A�*

epsilon�Nl��dK.       ��W�	E^̩�A�* 

Average reward per step�Nl�G��       ��2	�E^̩�A�*

epsilon�Nl�0(x.       ��W�	�R!^̩�A�* 

Average reward per step�Nl���Y�       ��2	7S!^̩�A�*

epsilon�Nl��a�C.       ��W�	�T#^̩�A�* 

Average reward per step�Nl��m�P       ��2	?U#^̩�A�*

epsilon�Nl�L1h�.       ��W�	9D%^̩�A�* 

Average reward per step�Nl���E       ��2	�D%^̩�A�*

epsilon�Nl�:��.       ��W�	�l'^̩�A�* 

Average reward per step�Nl�3���       ��2	"m'^̩�A�*

epsilon�Nl����.       ��W�	�n)^̩�A�* 

Average reward per step�Nl���8)       ��2	�o)^̩�A�*

epsilon�Nl��A��.       ��W�	�v+^̩�A�* 

Average reward per step�Nl�eQ�       ��2	dw+^̩�A�*

epsilon�Nl���.       ��W�	]�-^̩�A�* 

Average reward per step�Nl�s��       ��2		�-^̩�A�*

epsilon�Nl�ڀ��.       ��W�	./^̩�A�* 

Average reward per step�Nl����       ��2	�/^̩�A�*

epsilon�Nl�׋�=.       ��W�	_�1^̩�A�* 

Average reward per step�Nl�-�z       ��2	��1^̩�A�*

epsilon�Nl�L��.       ��W�	�<3^̩�A�* 

Average reward per step�Nl�Yד_       ��2	h=3^̩�A�*

epsilon�Nl�X�[N.       ��W�	Vc5^̩�A�* 

Average reward per step�Nl��4f       ��2	Ad5^̩�A�*

epsilon�Nl����E.       ��W�	Zg7^̩�A�* 

Average reward per step�Nl�go       ��2	Zh7^̩�A�*

epsilon�Nl��� .       ��W�	Sy9^̩�A�* 

Average reward per step�Nl����)       ��2	z9^̩�A�*

epsilon�Nl��nH	.       ��W�	�~;^̩�A�* 

Average reward per step�Nl����       ��2	�;^̩�A�*

epsilon�Nl�!��1.       ��W�	g�=^̩�A�* 

Average reward per step�Nl��v^E       ��2	��=^̩�A�*

epsilon�Nl��/�.       ��W�	�?^̩�A�* 

Average reward per step�Nl����       ��2	��?^̩�A�*

epsilon�Nl�ӕE.       ��W�	 zB^̩�A�* 

Average reward per step�Nl����O       ��2	�zB^̩�A�*

epsilon�Nl���.       ��W�	P4F^̩�A�* 

Average reward per step�Nl�v�F�       ��2	&5F^̩�A�*

epsilon�Nl��:�.       ��W�	u�G^̩�A�* 

Average reward per step�Nl�(f�	       ��2	K�G^̩�A�*

epsilon�Nl�"��4.       ��W�	��I^̩�A�* 

Average reward per step�Nl��a       ��2	/�I^̩�A�*

epsilon�Nl��h�.       ��W�	��K^̩�A�* 

Average reward per step�Nl���       ��2	��K^̩�A�*

epsilon�Nl�����.       ��W�	��M^̩�A�* 

Average reward per step�Nl���0V       ��2	��M^̩�A�*

epsilon�Nl��:i0       ���_	�N^̩�A*#
!
Average reward per episode��{���5�.       ��W�	�N^̩�A*!

total reward per episode  x���8�.       ��W�	tBR^̩�A�* 

Average reward per step��{��o]       ��2	JCR^̩�A�*

epsilon��{���1.       ��W�	�LT^̩�A�* 

Average reward per step��{����s       ��2	QMT^̩�A�*

epsilon��{�E�j.       ��W�	��V^̩�A�* 

Average reward per step��{�7�i�       ��2	��V^̩�A�*

epsilon��{�@��.       ��W�	X^̩�A�* 

Average reward per step��{�Ku       ��2	�X^̩�A�*

epsilon��{��>�>.       ��W�	)AZ^̩�A�* 

Average reward per step��{�{X �       ��2	�AZ^̩�A�*

epsilon��{��m:�.       ��W�	�B\^̩�A�* 

Average reward per step��{����       ��2	�C\^̩�A�*

epsilon��{�ӹ�.       ��W�	�_^̩�A�* 

Average reward per step��{�F�.G       ��2	�_^̩�A�*

epsilon��{�ﲦ-.       ��W�	c`^̩�A�* 

Average reward per step��{���       ��2	�c`^̩�A�*

epsilon��{��GVq.       ��W�	Ίb^̩�A�* 

Average reward per step��{���;�       ��2	i�b^̩�A�*

epsilon��{�%:.       ��W�	��d^̩�A�* 

Average reward per step��{�| ��       ��2	��d^̩�A�*

epsilon��{���w.       ��W�	s�f^̩�A�* 

Average reward per step��{�{<.P       ��2	I�f^̩�A�*

epsilon��{����4.       ��W�	_
i^̩�A�* 

Average reward per step��{�l���       ��2	�
i^̩�A�*

epsilon��{�+�.       ��W�	�2k^̩�A�* 

Average reward per step��{�xe��       ��2	�3k^̩�A�*

epsilon��{��9�`.       ��W�	��l^̩�A�* 

Average reward per step��{�eV�s       ��2	��l^̩�A�*

epsilon��{��Mo�.       ��W�	��n^̩�A�* 

Average reward per step��{�!�]�       ��2	v�n^̩�A�*

epsilon��{�'��[.       ��W�	d�p^̩�A�* 

Average reward per step��{��Ӗ4       ��2	�p^̩�A�*

epsilon��{��:fh.       ��W�	�s^̩�A�* 

Average reward per step��{�
�֪       ��2	Ms^̩�A�*

epsilon��{�]�:.       ��W�	� u^̩�A�* 

Average reward per step��{�X��       ��2	O!u^̩�A�*

epsilon��{�67�.       ��W�	�9w^̩�A�* 

Average reward per step��{�z[-�       ��2	`:w^̩�A�*

epsilon��{��Y�0       ���_	7Tw^̩�A*#
!
Average reward per episode�k��D�.       ��W�	�Tw^̩�A*!

total reward per episode  "��N�.       ��W�	�c{^̩�A�* 

Average reward per step�k�Q�8�       ��2	�d{^̩�A�*

epsilon�k�xoK.       ��W�	Է}^̩�A�* 

Average reward per step�k�����       ��2	s�}^̩�A�*

epsilon�k��U23.       ��W�	k^̩�A�* 

Average reward per step�k��5��       ��2	E^̩�A�*

epsilon�k���.       ��W�	�ʁ^̩�A�* 

Average reward per step�k��-w       ��2	6́^̩�A�*

epsilon�k�=���.       ��W�	�=�^̩�A�* 

Average reward per step�k�_? �       ��2	�>�^̩�A�*

epsilon�k��I��.       ��W�	ς�^̩�A�* 

Average reward per step�k���t�       ��2	k��^̩�A�*

epsilon�k��F�.       ��W�	꓇^̩�A�* 

Average reward per step�k�}��f       ��2	���^̩�A�*

epsilon�k��lu.       ��W�	���^̩�A�* 

Average reward per step�k��59       ��2	o��^̩�A�*

epsilon�k���.       ��W�	��^̩�A�* 

Average reward per step�k���ۀ       ��2	L�^̩�A�*

epsilon�k�$�t�.       ��W�	�8�^̩�A�* 

Average reward per step�k���9       ��2	q9�^̩�A�*

epsilon�k���4.       ��W�	�^̩�A�* 

Average reward per step�k���       ��2	b��^̩�A�*

epsilon�k�t��.       ��W�	)��^̩�A�* 

Average reward per step�k��&��       ��2	$��^̩�A�*

epsilon�k���CK.       ��W�	'��^̩�A�* 

Average reward per step�k���-�       ��2	���^̩�A�*

epsilon�k���$�.       ��W�	�ҕ^̩�A�* 

Average reward per step�k�%u]-       ��2	�ӕ^̩�A�*

epsilon�k�՜�.       ��W�	gї^̩�A�* 

Average reward per step�k�b{ɼ       ��2	_җ^̩�A�*

epsilon�k���".       ��W�	ٙ^̩�A�* 

Average reward per step�k�AM��       ��2	�ٙ^̩�A�*

epsilon�k�%�..       ��W�	�^̩�A�* 

Average reward per step�k��(A\       ��2	��^̩�A�*

epsilon�k�di�.       ��W�	�^̩�A�* 

Average reward per step�k�,���       ��2	��^̩�A�*

epsilon�k�/�\�.       ��W�	��^̩�A�* 

Average reward per step�k����       ��2	��^̩�A�*

epsilon�k���o.       ��W�	S��^̩�A�* 

Average reward per step�k��I8       ��2	)��^̩�A�*

epsilon�k�xO4.       ��W�	ܣ^̩�A�* 

Average reward per step�k����3       ��2	�ܣ^̩�A�*

epsilon�k�}m�].       ��W�	�^̩�A�* 

Average reward per step�k�ƹ\�       ��2	��^̩�A�*

epsilon�k�G�ճ.       ��W�	k�^̩�A�* 

Average reward per step�k��       ��2	R�^̩�A�*

epsilon�k�4.       ��W�	�+�^̩�A�* 

Average reward per step�k�$�B       ��2	�,�^̩�A�*

epsilon�k��!�.       ��W�	�M�^̩�A�* 

Average reward per step�k�����       ��2	DN�^̩�A�*

epsilon�k��R).       ��W�	Y��^̩�A�* 

Average reward per step�k�+; X       ��2	��^̩�A�*

epsilon�k���.       ��W�	�$�^̩�A�* 

Average reward per step�k��E
9       ��2	�%�^̩�A�*

epsilon�k���!>.       ��W�	��^̩�A�* 

Average reward per step�k���]       ��2	���^̩�A�*

epsilon�k�H$��.       ��W�	�ӳ^̩�A�* 

Average reward per step�k�A]��       ��2	�Գ^̩�A�*

epsilon�k��l�.       ��W�	���^̩�A�* 

Average reward per step�k���]       ��2	���^̩�A�*

epsilon�k�[���.       ��W�	���^̩�A�* 

Average reward per step�k�'l�U       ��2	e¸^̩�A�*

epsilon�k�$�E�.       ��W�	��^̩�A�* 

Average reward per step�k��9�       ��2	_�^̩�A�*

epsilon�k����.       ��W�	c�^̩�A�* 

Average reward per step�k�z��|       ��2	Z�^̩�A�*

epsilon�k��O�(.       ��W�	� �^̩�A�* 

Average reward per step�k�@>��       ��2	��^̩�A�*

epsilon�k�Dt�f0       ���_	F"�^̩�A*#
!
Average reward per episode��g�͵6�.       ��W�	�"�^̩�A*!

total reward per episode  ���ah.       ��W�	���^̩�A�* 

Average reward per step��g���	       ��2	���^̩�A�*

epsilon��g���i.       ��W�	��^̩�A�* 

Average reward per step��g�30�~       ��2	e�^̩�A�*

epsilon��g��%J�.       ��W�	'��^̩�A�* 

Average reward per step��g���~�       ��2	���^̩�A�*

epsilon��g�{~.       ��W�	>��^̩�A�* 

Average reward per step��g���?�       ��2	%��^̩�A�*

epsilon��g�	�..       ��W�	��^̩�A�* 

Average reward per step��g��C�%       ��2	��^̩�A�*

epsilon��g��詴.       ��W�	��^̩�A�* 

Average reward per step��g�Μz       ��2	� �^̩�A�*

epsilon��g���P.       ��W�	�^̩�A�* 

Average reward per step��g�qav       ��2	��^̩�A�*

epsilon��g�`�=4.       ��W�	P4�^̩�A�* 

Average reward per step��g�5)��       ��2	5�^̩�A�*

epsilon��g�̫)a.       ��W�	 ��^̩�A�* 

Average reward per step��g��D       ��2	���^̩�A�*

epsilon��g�c��.       ��W�	r��^̩�A�* 

Average reward per step��g��<��       ��2	D��^̩�A�*

epsilon��g���s .       ��W�	��^̩�A�* 

Average reward per step��g�z{�s       ��2	��^̩�A�*

epsilon��g�(�L.       ��W�	���^̩�A�* 

Average reward per step��g����\       ��2	���^̩�A�*

epsilon��g�?\ .       ��W�	�(�^̩�A�* 

Average reward per step��g���       ��2	�)�^̩�A�*

epsilon��g��;/.       ��W�	`��^̩�A�* 

Average reward per step��g�D��k       ��2	-��^̩�A�*

epsilon��g�����.       ��W�	�h�^̩�A�* 

Average reward per step��g�����       ��2	Qi�^̩�A�*

epsilon��g�[���.       ��W�	s��^̩�A�* 

Average reward per step��g��d��       ��2	Z��^̩�A�*

epsilon��g�n�!.       ��W�	���^̩�A�* 

Average reward per step��g��h��       ��2	l��^̩�A�*

epsilon��g�I&.       ��W�	���^̩�A�* 

Average reward per step��g��2       ��2	���^̩�A�*

epsilon��g���A.       ��W�	f��^̩�A�* 

Average reward per step��g���"       ��2	]��^̩�A�*

epsilon��g��È�.       ��W�	���^̩�A�* 

Average reward per step��g����u       ��2	Q��^̩�A�*

epsilon��g�UB`�.       ��W�	���^̩�A�* 

Average reward per step��g��s       ��2	���^̩�A�*

epsilon��g�X��.       ��W�	-��^̩�A�* 

Average reward per step��g��IWo       ��2	���^̩�A�*

epsilon��g�e8�s.       ��W�	���^̩�A�* 

Average reward per step��g��4�       ��2	���^̩�A�*

epsilon��g��M��.       ��W�	g�^̩�A�* 

Average reward per step��g�1�j       ��2	h�^̩�A�*

epsilon��g�Q*o.       ��W�	� �^̩�A�* 

Average reward per step��g��       ��2	~�^̩�A�*

epsilon��g�����.       ��W�	�
�^̩�A�* 

Average reward per step��g��ѳy       ��2	��^̩�A�*

epsilon��g�:�L.       ��W�	�1�^̩�A�* 

Average reward per step��g�ز�       ��2	H2�^̩�A�*

epsilon��g���d.       ��W�	!��^̩�A�* 

Average reward per step��g��?X.       ��2	���^̩�A�*

epsilon��g���c�.       ��W�	��^̩�A�* 

Average reward per step��g�4y|=       ��2	��^̩�A�*

epsilon��g�ʧ{�.       ��W�	��^̩�A�* 

Average reward per step��g�8�l�       ��2	��^̩�A�*

epsilon��g�M�.       ��W�	��^̩�A�* 

Average reward per step��g��+��       ��2	���^̩�A�*

epsilon��g�d8�#.       ��W�	v5_̩�A�* 

Average reward per step��g����E       ��2	&6_̩�A�*

epsilon��g��Tv3.       ��W�	s�_̩�A�* 

Average reward per step��g��`*       ��2	�_̩�A�*

epsilon��g��R�C.       ��W�	�_̩�A�* 

Average reward per step��g���       ��2	�_̩�A�*

epsilon��g���x�.       ��W�	��_̩�A�* 

Average reward per step��g��h�       ��2	��_̩�A�*

epsilon��g�wv'.       ��W�	�5_̩�A�* 

Average reward per step��g���24       ��2	Y6_̩�A�*

epsilon��g�W��.       ��W�	�-	_̩�A�* 

Average reward per step��g��y��       ��2	�.	_̩�A�*

epsilon��g�i�i�.       ��W�	CX_̩�A�* 

Average reward per step��g�O���       ��2	Y_̩�A�*

epsilon��g�����.       ��W�		�_̩�A�* 

Average reward per step��g�"��?       ��2	��_̩�A�*

epsilon��g�V��.       ��W�	/�_̩�A�* 

Average reward per step��g�x	�       ��2	��_̩�A�*

epsilon��g�:~0       ���_	y_̩�A*#
!
Average reward per episode��4�,��.       ��W�	%_̩�A*!

total reward per episode  ��Ī/�.       ��W�	�G_̩�A�* 

Average reward per step��4�k�       ��2	gH_̩�A�*

epsilon��4����.       ��W�	�_̩�A�* 

Average reward per step��4�L�:�       ��2	̸_̩�A�*

epsilon��4�
�w�.       ��W�	��_̩�A�* 

Average reward per step��4����        ��2	p�_̩�A�*

epsilon��4���p.       ��W�	|�_̩�A�* 

Average reward per step��4�H�lt       ��2	�_̩�A�*

epsilon��4�uS�.       ��W�	R�_̩�A�* 

Average reward per step��4����       ��2	$�_̩�A�*

epsilon��4��J��.       ��W�	��_̩�A�* 

Average reward per step��4�����       ��2	F�_̩�A�*

epsilon��4�t�.       ��W�	u�_̩�A�* 

Average reward per step��4���u       ��2	G�_̩�A�*

epsilon��4�����.       ��W�	�� _̩�A�* 

Average reward per step��4�E��       ��2	�� _̩�A�*

epsilon��4����w.       ��W�	��"_̩�A�* 

Average reward per step��4�M��o       ��2	��"_̩�A�*

epsilon��4���$.       ��W�	n�$_̩�A�* 

Average reward per step��4�����       ��2	"�$_̩�A�*

epsilon��4��X�.       ��W�	��&_̩�A�* 

Average reward per step��4�}��b       ��2	3�&_̩�A�*

epsilon��4�\���.       ��W�	�(_̩�A�* 

Average reward per step��4��x��       ��2	��(_̩�A�*

epsilon��4�#i�g.       ��W�	d�)_̩�A�* 

Average reward per step��4��)��       ��2	m�)_̩�A�*

epsilon��4���.       ��W�	n�+_̩�A�* 

Average reward per step��4�8F/`       ��2	&�+_̩�A�*

epsilon��4��`�}.       ��W�	2�-_̩�A�* 

Average reward per step��4��/�       ��2	��-_̩�A�*

epsilon��4��k�.       ��W�	L�/_̩�A�* 

Average reward per step��4�_�8       ��2	��/_̩�A�*

epsilon��4�#���.       ��W�	�1_̩�A�* 

Average reward per step��4�X�SQ       ��2	��1_̩�A�*

epsilon��4�*Pj�.       ��W�	)3_̩�A�* 

Average reward per step��4�ㅐ       ��2	�3_̩�A�*

epsilon��4�N0+�.       ��W�	��4_̩�A�* 

Average reward per step��4��W       ��2	A�4_̩�A�*

epsilon��4���@U.       ��W�	��6_̩�A�* 

Average reward per step��4�#�q%       ��2	v�6_̩�A�*

epsilon��4���lL.       ��W�	��8_̩�A�* 

Average reward per step��4�d	��       ��2	~�8_̩�A�*

epsilon��4���6�.       ��W�	��:_̩�A�* 

Average reward per step��4���,�       ��2	G�:_̩�A�*

epsilon��4�Ɣx.       ��W�	Q2<_̩�A�* 

Average reward per step��4���n       ��2	�4<_̩�A�*

epsilon��4���b;.       ��W�	�0>_̩�A�* 

Average reward per step��4���T
       ��2	�1>_̩�A�*

epsilon��4����h.       ��W�	�/@_̩�A�* 

Average reward per step��4��Pte       ��2	�0@_̩�A�*

epsilon��4��v�(.       ��W�	d B_̩�A�* 

Average reward per step��4��	��       ��2	� B_̩�A�*

epsilon��4��͋4.       ��W�	?D_̩�A�* 

Average reward per step��4���c       ��2	D_̩�A�*

epsilon��4���X.       ��W�	�F_̩�A�* 

Average reward per step��4�3=Lp       ��2	�F_̩�A�*

epsilon��4����m.       ��W�	A*H_̩�A�* 

Average reward per step��4��5gl       ��2	�*H_̩�A�*

epsilon��4�ױ��.       ��W�	�uI_̩�A�* 

Average reward per step��4��]�       ��2	�vI_̩�A�*

epsilon��4��5�0       ���_	_�I_̩�A*#
!
Average reward per episode  �����.       ��W�	�I_̩�A*!

total reward per episode  ��<FA.       ��W�	[|M_̩�A�* 

Average reward per step  ���Λ�       ��2	}M_̩�A�*

epsilon  ���+�.       ��W�	anO_̩�A�* 

Average reward per step  ����       ��2	�nO_̩�A�*

epsilon  ���S�.       ��W�	l_Q_̩�A�* 

Average reward per step  ��K�1�       ��2	`Q_̩�A�*

epsilon  ��רJ.       ��W�	QiS_̩�A�* 

Average reward per step  ����	       ��2	jS_̩�A�*

epsilo