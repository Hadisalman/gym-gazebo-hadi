       �K"	  @�˩�Abrain.Event:2^s쯛�      �*	��@�˩�A"��
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
flatten_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
new_axis_mask *
end_mask*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask 
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
dtype0*
_output_shapes
:*
valueB"      
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
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0
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
dtype0*
_output_shapes
:*
valueB"   d   
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
VariableV2*
dtype0*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0*!
_class
loc:@dense_2/kernel
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
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
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������d*
T0
\
activation_2/ReluReludense_2/BiasAdd*'
_output_shapes
:���������d*
T0
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
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0*!
_class
loc:@dense_3/kernel
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:2*
	container *
shape:2
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
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
_output_shapes

:2*
seed2��K*
seed���)*
T0*
dtype0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
T0*
_output_shapes
: 
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
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
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
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�m?
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
_output_shapes

:*
	container *
shape
:*
shared_name *
dtype0
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
dense_5/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
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
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
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
:���������*
T0*
Index0
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
 lambda_1/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
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
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name 
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
dtype0*
_output_shapes
:*
valueB"   d   
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
VariableV2*
_output_shapes
:	�d*
	container *
shape:	�d*
shared_name *
dtype0
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
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
use_locking(*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d
w
dense_2_1/bias/readIdentitydense_2_1/bias*
_output_shapes
:d*
T0*!
_class
loc:@dense_2_1/bias
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
�
dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*'
_output_shapes
:���������d*
T0*
data_formatNHWC
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
dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
T0*
_output_shapes
: 
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
VariableV2*
shape
:d2*
shared_name *
dtype0*
_output_shapes

:d2*
	container 
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
dense_3_1/ConstConst*
_output_shapes
:2*
valueB2*    *
dtype0
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
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*'
_output_shapes
:���������2*
transpose_a( *
transpose_b( *
T0
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
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
T0*
_output_shapes

:2
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
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
dense_5_1/kernel/readIdentitydense_5_1/kernel*
_output_shapes

:*
T0*#
_class
loc:@dense_5_1/kernel
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
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:���������
d
lambda_1_1/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
"lambda_1_1/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
s
"lambda_1_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*
T0*'
_output_shapes
:���������
q
 lambda_1_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB"       
s
"lambda_1_1/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
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
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_2/kernel
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_output_shapes
: *
_class
loc:@dense_2/bias*
dtype0
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
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*
_output_shapes
: *!
_class
loc:@dense_5_1/bias*
dtype0
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
dtype0*
_output_shapes
:	�*
shape:	�
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
Assign_3Assigndense_2_1/biasPlaceholder_3*
validate_shape(*
_output_shapes
:d*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias
^
Placeholder_4Placeholder*
_output_shapes

:d2*
shape
:d2*
dtype0
�
Assign_4Assigndense_3_1/kernelPlaceholder_4*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking( 
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
dtype0*
_output_shapes

:2*
shape
:2
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
SGD/lr/initial_valueConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: 
j
SGD/lr
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
SGD/momentum/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: *
use_locking(
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
lambda_1_targetPlaceholder*0
_output_shapes
:������������������*%
shape:������������������*
dtype0
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
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0	
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
SGD_1/lr/initial_valueConst*
_output_shapes
: *
valueB
 *
�#<*
dtype0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
SGD_1/decay/AssignAssignSGD_1/decaySGD_1/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD_1/decay*
validate_shape(*
_output_shapes
: 
j
SGD_1/decay/readIdentitySGD_1/decay*
_output_shapes
: *
T0*
_class
loc:@SGD_1/decay
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
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
loss_2/mul_2Mulloss_2/Selectmask*'
_output_shapes
:���������*
T0
g
loss_2/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
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
loss_3/loss_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*#
_output_shapes
:���������*
T0
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
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:���������
b
loss_3/loss_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
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
loss_3/lambda_1_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*
T0*'
_output_shapes
:���������
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
metrics_2/mean_q/MeanMeanmetrics_2/mean_q/Maxmetrics_2/mean_q/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
IsVariableInitialized_30IsVariableInitializedSGD_1/lr*
dtype0*
_output_shapes
: *
_class
loc:@SGD_1/lr
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
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"�1�>F      ��w	'PB�˩�AJ��
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
:*
T0*
Index0
Y
flatten_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
~
flatten_1/ProdProdflatten_1/strided_sliceflatten_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes
:	�*
T0
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
dense_1/ConstConst*
_output_shapes	
:�*
valueB�*    *
dtype0
z
dense_1/bias
VariableV2*
_output_shapes	
:�*
	container *
shape:�*
shared_name *
dtype0
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
VariableV2*
shared_name *
dtype0*
_output_shapes

:d2*
	container *
shape
:d2
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
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
T0*
_class
loc:@dense_3/bias*
validate_shape(*
_output_shapes
:2*
use_locking(
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:2*
T0*
_class
loc:@dense_3/bias
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
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
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
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
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
dense_5/kernel/readIdentitydense_5/kernel*!
_class
loc:@dense_5/kernel*
_output_shapes

:*
T0
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
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:*
use_locking(
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
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
end_mask*#
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
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
dtype0*
_output_shapes
:*
valueB"       
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
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask 
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
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
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
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
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
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1*
validate_shape(
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
Adam/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
n

Adam/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
flatten_1_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
k
!flatten_1_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
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
flatten_1_1/ReshapeReshapeflatten_1_input_1flatten_1_1/stack*
Tshape0*0
_output_shapes
:������������������*
T0
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
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	�*
seed2��h*
seed���)
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
VariableV2*
dtype0*
_output_shapes
:	�*
	container *
shape:	�*
shared_name 
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
dense_2_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *?�ʽ*
dtype0
a
dense_2_1/random_uniform/maxConst*
valueB
 *?��=*
dtype0*
_output_shapes
: 
�
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
_output_shapes
:	�d*
seed2Ӽ�*
seed���)*
T0*
dtype0
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
dense_2_1/kernel/readIdentitydense_2_1/kernel*#
_class
loc:@dense_2_1/kernel*
_output_shapes
:	�d*
T0
\
dense_2_1/ConstConst*
_output_shapes
:d*
valueBd*    *
dtype0
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
dense_2_1/bias/AssignAssigndense_2_1/biasdense_2_1/Const*
use_locking(*
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d
w
dense_2_1/bias/readIdentitydense_2_1/bias*
T0*!
_class
loc:@dense_2_1/bias*
_output_shapes
:d
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( *
T0
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
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:d2*
seed2���*
seed���)
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
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0
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
dense_4_1/kernel/AssignAssigndense_4_1/kerneldense_4_1/random_uniform*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2*
use_locking(*
T0
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
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_4_1/bias/AssignAssigndense_4_1/biasdense_4_1/Const*!
_class
loc:@dense_4_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
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
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2��*
seed���)
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
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
dense_5_1/kernel/readIdentitydense_5_1/kernel*
T0*#
_class
loc:@dense_5_1/kernel*
_output_shapes

:
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
shrink_axis_mask*

begin_mask*
ellipsis_mask *
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
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask 
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*'
_output_shapes
:���������*
T0
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
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_output_shapes
: *
_class
loc:@dense_1/bias*
dtype0
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_2/kernel
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
IsVariableInitialized_11IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
_output_shapes
: *
_class
loc:@Adam/beta_1*
dtype0
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
IsVariableInitialized_21IsVariableInitializeddense_4_1/kernel*
_output_shapes
: *#
_class
loc:@dense_4_1/kernel*
dtype0
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
Assign_2Assigndense_2_1/kernelPlaceholder_2*
validate_shape(*
_output_shapes
:	�d*
use_locking( *
T0*#
_class
loc:@dense_2_1/kernel
V
Placeholder_3Placeholder*
shape:d*
dtype0*
_output_shapes
:d
�
Assign_3Assigndense_2_1/biasPlaceholder_3*!
_class
loc:@dense_2_1/bias*
validate_shape(*
_output_shapes
:d*
use_locking( *
T0
^
Placeholder_4Placeholder*
dtype0*
_output_shapes

:d2*
shape
:d2
�
Assign_4Assigndense_3_1/kernelPlaceholder_4*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking( *
T0
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
Assign_7Assigndense_4_1/biasPlaceholder_7*
T0*!
_class
loc:@dense_4_1/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
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
Placeholder_9Placeholder*
shape:*
dtype0*
_output_shapes
:
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
shape: *
shared_name *
dtype0	*
_output_shapes
: *
	container 
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
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
[
SGD/lr/readIdentitySGD/lr*
_output_shapes
: *
T0*
_class
loc:@SGD/lr
_
SGD/momentum/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
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
SGD/momentum/readIdentitySGD/momentum*
T0*
_class
loc:@SGD/momentum*
_output_shapes
: 
\
SGD/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
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
SGD/decay/readIdentity	SGD/decay*
_class
loc:@SGD/decay*
_output_shapes
: *
T0
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
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*'
_output_shapes
:���������*
T0
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
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
�
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
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
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
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
SGD_1/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
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
SGD_1/iterations/AssignAssignSGD_1/iterationsSGD_1/iterations/initial_value*
use_locking(*
T0	*#
_class
loc:@SGD_1/iterations*
validate_shape(*
_output_shapes
: 
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
SGD_1/lr/readIdentitySGD_1/lr*
_output_shapes
: *
T0*
_class
loc:@SGD_1/lr
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
SGD_1/decay/readIdentitySGD_1/decay*
_output_shapes
: *
T0*
_class
loc:@SGD_1/decay
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
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*'
_output_shapes
:���������*
T0
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
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*#
_output_shapes
:���������*

DstT0*

SrcT0

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
loss_2/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
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
loss_2/sub_1Subloss_2/Abs_1loss_2/sub_1/y*
T0*'
_output_shapes
:���������
S
loss_2/mul_1/xConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*
T0*'
_output_shapes
:���������
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
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*#
_output_shapes
:���������*
T0
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
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:���������
b
loss_3/loss_loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
loss_3/lambda_1_loss/zeros_like	ZerosLikelambda_1/sub*'
_output_shapes
:���������*
T0
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
�
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
�
loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*#
_output_shapes
:���������*
T0
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
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
q
&metrics_2/mean_q/Max/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*#
_output_shapes
:���������*

Tidx0*
	keep_dims( *
T0
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
metrics_2/mean_q/Const_1Const*
_output_shapes
: *
valueB *
dtype0
�
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
IsVariableInitialized_25IsVariableInitializedSGD/iterations*
_output_shapes
: *!
_class
loc:@SGD/iterations*
dtype0	
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
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0�Ti.       ��W�	�Q3�˩�A*#
!
Average reward per episode���fh�,       ���E	]R3�˩�A*!

total reward per episode  �u��_-       <A��	J(7�˩�A* 

Average reward per step����       `/�#	E)7�˩�A*

epsilon���_y{-       <A��	0c8�˩�A* 

Average reward per step������X       `/�#	�c8�˩�A*

epsilon����m��-       <A��	�;:�˩�A* 

Average reward per step���(��z       `/�#	y<:�˩�A*

epsilon����'~p-       <A��	�l;�˩�A * 

Average reward per step���́��       `/�#	�m;�˩�A *

epsilon���k�A�-       <A��	�N=�˩�A!* 

Average reward per step�������       `/�#	QO=�˩�A!*

epsilon���
�-       <A��	��>�˩�A"* 

Average reward per step�����$%       `/�#	L�>�˩�A"*

epsilon���d��-       <A��	��@�˩�A#* 

Average reward per step������:       `/�#	!�@�˩�A#*

epsilon���rMt�-       <A��	��B�˩�A$* 

Average reward per step���Dq�       `/�#	ǃB�˩�A$*

epsilon���b��-       <A��	�bD�˩�A%* 

Average reward per step���'tN�       `/�#	�cD�˩�A%*

epsilon���Ǜs-       <A��	T�E�˩�A&* 

Average reward per step���2�       `/�#	�E�˩�A&*

epsilon�����3-       <A��	��G�˩�A'* 

Average reward per step����g�       `/�#	l�G�˩�A'*

epsilon����U�F-       <A��	@�I�˩�A(* 

Average reward per step���`ޯ�       `/�#	ۈI�˩�A(*

epsilon����c��-       <A��	F^K�˩�A)* 

Average reward per step����R�       `/�#	_K�˩�A)*

epsilon�������-       <A��	V�L�˩�A** 

Average reward per step����]��       `/�#	4�L�˩�A**

epsilon������ -       <A��	ʋN�˩�A+* 

Average reward per step���A�k       `/�#	e�N�˩�A+*

epsilon����F�-       <A��	ӿO�˩�A,* 

Average reward per step�����C�       `/�#	��O�˩�A,*

epsilon������W-       <A��	�Q�˩�A-* 

Average reward per step���E]�       `/�#	��Q�˩�A-*

epsilon���xX�X-       <A��	��S�˩�A.* 

Average reward per step�����j       `/�#	S�˩�A.*

epsilon����z-       <A��	ԸT�˩�A/* 

Average reward per step���Y�K?       `/�#	��T�˩�A/*

epsilon����*��-       <A��	��V�˩�A0* 

Average reward per step����(i       `/�#	h�V�˩�A0*

epsilon����'-       <A��	�W�˩�A1* 

Average reward per step���d>^X       `/�#	��W�˩�A1*

epsilon���D�+�-       <A��	�Y�˩�A2* 

Average reward per step���?qj�       `/�#	��Y�˩�A2*

epsilon���.��T-       <A��	��[�˩�A3* 

Average reward per step���[G       `/�#	1�[�˩�A3*

epsilon���0��0       ���_	��[�˩�A*#
!
Average reward per episoded!��Zɺ.       ��W�	5�[�˩�A*!

total reward per episode  ����-       <A��	e�^�˩�A4* 

Average reward per stepd!���E��       `/�#		�^�˩�A4*

epsilond!���\�B-       <A��	�`�˩�A5* 

Average reward per stepd!���ŷ�       `/�#	P�`�˩�A5*

epsilond!���s
 -       <A��	&�b�˩�A6* 

Average reward per stepd!���cr�       `/�#	��b�˩�A6*

epsilond!��%���-       <A��	�c�˩�A7* 

Average reward per stepd!��Â�       `/�#	��c�˩�A7*

epsilond!���k4-       <A��	R�e�˩�A8* 

Average reward per stepd!��GRyv       `/�#	�e�˩�A8*

epsilond!��&�^�-       <A��	��f�˩�A9* 

Average reward per stepd!��s�H�       `/�#	N�f�˩�A9*

epsilond!��EP-       <A��	��h�˩�A:* 

Average reward per stepd!���̾�       `/�#	��h�˩�A:*

epsilond!��'���-       <A��	 �j�˩�A;* 

Average reward per stepd!�����!       `/�#	��j�˩�A;*

epsilond!����-       <A��	�k�˩�A<* 

Average reward per stepd!����{       `/�#	U�k�˩�A<*

epsilond!��v��-       <A��	�m�˩�A=* 

Average reward per stepd!��PQ�       `/�#	��m�˩�A=*

epsilond!��7T�-       <A��	�o�˩�A>* 

Average reward per stepd!����"a       `/�#	go�˩�A>*

epsilond!���4-       <A��	V�p�˩�A?* 

Average reward per stepd!��.lw       `/�#	(�p�˩�A?*

epsilond!�����-       <A��	�r�˩�A@* 

Average reward per stepd!����DL       `/�#	��r�˩�A@*

epsilond!���c�4-       <A��	S!t�˩�AA* 

Average reward per stepd!��9$�       `/�#	2"t�˩�AA*

epsilond!��9^2w-       <A��	qv�˩�AB* 

Average reward per stepd!����<       `/�#	�v�˩�AB*

epsilond!����cf-       <A��	�:w�˩�AC* 

Average reward per stepd!����
�       `/�#	�;w�˩�AC*

epsilond!��&~��-       <A��	Xy�˩�AD* 

Average reward per stepd!���ENS       `/�#	iy�˩�AD*

epsilond!���p�-       <A��	��z�˩�AE* 

Average reward per stepd!��*YL       `/�#	��z�˩�AE*

epsilond!��m��-       <A��	�0|�˩�AF* 

Average reward per stepd!��
#�f       `/�#	�1|�˩�AF*

epsilond!��i��O-       <A��	,~�˩�AG* 

Average reward per stepd!���^��       `/�#	�~�˩�AG*

epsilond!�����-       <A��	�P�˩�AH* 

Average reward per stepd!������       `/�#	�Q�˩�AH*

epsilond!�� ��.-       <A��	YM��˩�AI* 

Average reward per stepd!����I       `/�#	N��˩�AI*

epsilond!����-�-       <A��	�4��˩�AJ* 

Average reward per stepd!�����       `/�#	?5��˩�AJ*

epsilond!�����]-       <A��	��˩�AK* 

Average reward per stepd!��Hȓ�       `/�#	���˩�AK*

epsilond!����.-       <A��	j2��˩�AL* 

Average reward per stepd!����       `/�#	"3��˩�AL*

epsilond!���l-       <A��	�3��˩�AM* 

Average reward per stepd!�����       `/�#	@4��˩�AM*

epsilond!��j��-       <A��	�H��˩�AN* 

Average reward per stepd!���C�\       `/�#	�I��˩�AN*

epsilond!��"�--       <A��	�-��˩�AO* 

Average reward per stepd!��]�$�       `/�#	{.��˩�AO*

epsilond!���7��-       <A��	}���˩�AP* 

Average reward per stepd!��I��       `/�#	���˩�AP*

epsilond!��[�6�-       <A��	2w��˩�AQ* 

Average reward per stepd!��ѳ0       `/�#	�w��˩�AQ*

epsilond!���B��-       <A��	�U��˩�AR* 

Average reward per stepd!�����C       `/�#	6V��˩�AR*

epsilond!��r�+y-       <A��	T���˩�AS* 

Average reward per stepd!��&���       `/�#	���˩�AS*

epsilond!���Vy-       <A��	m��˩�AT* 

Average reward per stepd!��N��       `/�#	�m��˩�AT*

epsilond!���m�-       <A��	�S��˩�AU* 

Average reward per stepd!����\�       `/�#	vT��˩�AU*

epsilond!��V�ը-       <A��	w���˩�AV* 

Average reward per stepd!��Ns�       `/�#	���˩�AV*

epsilond!��֘��-       <A��	dx��˩�AW* 

Average reward per stepd!��=(��       `/�#	�x��˩�AW*

epsilond!���D-       <A��	�_��˩�AX* 

Average reward per stepd!���?�}       `/�#	�`��˩�AX*

epsilond!���%�-       <A��	����˩�AY* 

Average reward per stepd!���s��       `/�#	p���˩�AY*

epsilond!����G-       <A��	�u��˩�AZ* 

Average reward per stepd!�����       `/�#	�v��˩�AZ*

epsilond!��%-       <A��	X���˩�A[* 

Average reward per stepd!��Юb       `/�#	����˩�A[*

epsilond!��VД�-       <A��	��˩�A\* 

Average reward per stepd!��5Q�U       `/�#	����˩�A\*

epsilond!���b�-       <A��	$|��˩�A]* 

Average reward per stepd!���|K�       `/�#	�|��˩�A]*

epsilond!�� ��#-       <A��	�ť�˩�A^* 

Average reward per stepd!����,:       `/�#	Lƥ�˩�A^*

epsilond!��߃>�-       <A��	����˩�A_* 

Average reward per stepd!���|bg       `/�#	<���˩�A_*

epsilond!��y5`�-       <A��	����˩�A`* 

Average reward per stepd!��𦪁       `/�#	ۊ��˩�A`*

epsilond!��R�5-       <A��	2ɪ�˩�Aa* 

Average reward per stepd!��Wc
L       `/�#	ʪ�˩�Aa*

epsilond!��=���-       <A��	����˩�Ab* 

Average reward per stepd!���]�       `/�#	����˩�Ab*

epsilond!���	�-       <A��	���˩�Ac* 

Average reward per stepd!��pK�T       `/�#	`��˩�Ac*

epsilond!��Aˇ-       <A��	Cǯ�˩�Ad* 

Average reward per stepd!������       `/�#	ȯ�˩�Ad*

epsilond!��|p-       <A��	�
��˩�Ae* 

Average reward per stepd!��;-'�       `/�#	1��˩�Ae*

epsilond!������-       <A��	���˩�Af* 

Average reward per stepd!�����       `/�#	\��˩�Af*

epsilond!����h�-       <A��	P��˩�Ag* 

Average reward per stepd!���z�\       `/�#	���˩�Ag*

epsilond!��t���-       <A��	^��˩�Ah* 

Average reward per stepd!��(�V       `/�#	���˩�Ah*

epsilond!�����-       <A��	��˩�Ai* 

Average reward per stepd!���~c       `/�#	���˩�Ai*

epsilond!����|-       <A��	���˩�Aj* 

Average reward per stepd!���o$       `/�#	'��˩�Aj*

epsilond!���G�h-       <A��	����˩�Ak* 

Average reward per stepd!��O�9       `/�#	'���˩�Ak*

epsilond!��R^��-       <A��	�)��˩�Al* 

Average reward per stepd!���H��       `/�#	$*��˩�Al*

epsilond!���J�-       <A��	���˩�Am* 

Average reward per stepd!��iA��       `/�#	s��˩�Am*

epsilond!���i-       <A��	����˩�An* 

Average reward per stepd!���v|�       `/�#	I���˩�An*

epsilond!��,��-       <A��	/0��˩�Ao* 

Average reward per stepd!��w���       `/�#	�0��˩�Ao*

epsilond!���g��-       <A��	���˩�Ap* 

Average reward per stepd!����Q�       `/�#	r��˩�Ap*

epsilond!�����-       <A��	N��˩�Aq* 

Average reward per stepd!��p�]:       `/�#	�N��˩�Aq*

epsilond!��R�4�-       <A��	b1��˩�Ar* 

Average reward per stepd!���!�^       `/�#	2��˩�Ar*

epsilond!����4�-       <A��	���˩�As* 

Average reward per stepd!��5���       `/�#	�	��˩�As*

epsilond!���#�-       <A��	@M��˩�At* 

Average reward per stepd!����       `/�#	�M��˩�At*

epsilond!���!�t-       <A��	�,��˩�Au* 

Average reward per stepd!����       `/�#	�-��˩�Au*

epsilond!��vɁ�-       <A��	�~��˩�Av* 

Average reward per stepd!���       `/�#	g��˩�Av*

epsilond!�����-       <A��	Mf��˩�Aw* 

Average reward per stepd!��԰~�       `/�#	�f��˩�Aw*

epsilond!��ļ,�-       <A��	�D��˩�Ax* 

Average reward per stepd!��]3�       `/�#	9E��˩�Ax*

epsilond!���:-       <A��	�(��˩�Ay* 

Average reward per stepd!��P��<       `/�#	J)��˩�Ay*

epsilond!��qï!-       <A��	g��˩�Az* 

Average reward per stepd!��'���       `/�#	�g��˩�Az*

epsilond!�����-       <A��	9C��˩�A{* 

Average reward per stepd!��}ٺB       `/�#	�C��˩�A{*

epsilond!��	O]-       <A��	I���˩�A|* 

Average reward per stepd!���b�       `/�#	s���˩�A|*

epsilond!��8@X�-       <A��	�d��˩�A}* 

Average reward per stepd!��7O��       `/�#	ke��˩�A}*

epsilond!��=��-       <A��	]��˩�A~* 

Average reward per stepd!����q       `/�#	�]��˩�A~*

epsilond!���ZP�-       <A��	cB��˩�A* 

Average reward per stepd!���77�       `/�#	AD��˩�A*

epsilond!����َ.       ��W�	o~��˩�A�* 

Average reward per stepd!������       ��2	��˩�A�*

epsilond!�����.       ��W�	�^��˩�A�* 

Average reward per stepd!��R�݉       ��2	�_��˩�A�*

epsilond!��S��K.       ��W�	R���˩�A�* 

Average reward per stepd!��.$       ��2	���˩�A�*

epsilond!��x_�.       ��W�	����˩�A�* 

Average reward per stepd!��3�.       ��2	����˩�A�*

epsilond!��Pye�.       ��W�	f��˩�A�* 

Average reward per stepd!��ǣ�       ��2	�f��˩�A�*

epsilond!������.       ��W�	���˩�A�* 

Average reward per stepd!�����       ��2	����˩�A�*

epsilond!�����.       ��W�	F���˩�A�* 

Average reward per stepd!���<w       ��2	���˩�A�*

epsilond!����yG.       ��W�	_~��˩�A�* 

Average reward per stepd!���C��       ��2	0��˩�A�*

epsilond!��kC�0       ���_	����˩�A*#
!
Average reward per episode�����?Ȁ.       ��W�	4���˩�A*!

total reward per episode  $��s�.       ��W�	�&��˩�A�* 

Average reward per step�����|a�       ��2	J'��˩�A�*

epsilon����4�nu.       ��W�	V��˩�A�* 

Average reward per step����^�;�       ��2	��˩�A�*

epsilon����1��.       ��W�	B���˩�A�* 

Average reward per step�����4�       ��2	���˩�A�*

epsilon�����/�.       ��W�	M���˩�A�* 

Average reward per step����xI��       ��2	#���˩�A�*

epsilon����.��.       ��W�	%���˩�A�* 

Average reward per step����f+��       ��2	����˩�A�*

epsilon�����:M.       ��W�	���˩�A�* 

Average reward per step�����߱�       ��2	���˩�A�*

epsilon����81��.       ��W�	i���˩�A�* 

Average reward per step�����E�[       ��2	���˩�A�*

epsilon����9��.       ��W�	����˩�A�* 

Average reward per step�����M��       ��2	e���˩�A�*

epsilon����::��.       ��W�	M���˩�A�* 

Average reward per step����l��       ��2	����˩�A�*

epsilon������݋.       ��W�	D���˩�A�* 

Average reward per step����4~q       ��2	
���˩�A�*

epsilon����J*&.       ��W�	i���˩�A�* 

Average reward per step����[��       ��2	;���˩�A�*

epsilon����+�bT.       ��W�	���˩�A�* 

Average reward per step������y�       ��2	r��˩�A�*

epsilon�����u�.       ��W�	�#�˩�A�* 

Average reward per step����2��       ��2	�$�˩�A�*

epsilon����=��.       ��W�	���˩�A�* 

Average reward per step����f��       ��2	���˩�A�*

epsilon����1s�.       ��W�	���˩�A�* 

Average reward per step�����7�       ��2	���˩�A�*

epsilon����u��).       ��W�	-!
�˩�A�* 

Average reward per step����ִ]�       ��2	"
�˩�A�*

epsilon����E�.       ��W�	�
�˩�A�* 

Average reward per step����M�`        ��2	N�˩�A�*

epsilon������ܟ.       ��W�	^��˩�A�* 

Average reward per step���� ��D       ��2	,��˩�A�*

epsilon����p|D�.       ��W�	�A�˩�A�* 

Average reward per step�����,5{       ��2	�B�˩�A�*

epsilon����2���.       ��W�	8�˩�A�* 

Average reward per step����Sl$       ��2	�8�˩�A�*

epsilon����@8Q�.       ��W�	9'�˩�A�* 

Average reward per step�����f|�       ��2	(�˩�A�*

epsilon�����f^.       ��W�	�c�˩�A�* 

Average reward per step����OS7w       ��2	Id�˩�A�*

epsilon����R�;�.       ��W�	�H�˩�A�* 

Average reward per step����9!�       ��2	kI�˩�A�*

epsilon����d���.       ��W�	t&�˩�A�* 

Average reward per step�����>r       ��2	='�˩�A�*

epsilon������Yg.       ��W�	�b�˩�A�* 

Average reward per step����S�Ծ       ��2	d�˩�A�*

epsilon����Ezd.       ��W�	EG�˩�A�* 

Average reward per step����@�ǉ       ��2	
H�˩�A�*

epsilon����
�э.       ��W�	�.�˩�A�* 

Average reward per step����>o��       ��2	�/�˩�A�*

epsilon����� �.       ��W�	���˩�A�* 

Average reward per step����|\�       ��2	=��˩�A�*

epsilon����"mc�.       ��W�	qZ �˩�A�* 

Average reward per step����$�       ��2	[ �˩�A�*

epsilon����L�u0       ���_	�t �˩�A*#
!
Average reward per episode�=��(�.       ��W�	du �˩�A*!

total reward per episode   Á�X�.       ��W�	W>$�˩�A�* 

Average reward per step�=����9�       ��2	!?$�˩�A�*

epsilon�=���)�q.       ��W�	�}%�˩�A�* 

Average reward per step�=��?֠�       ��2	|~%�˩�A�*

epsilon�=����K�.       ��W�	�\'�˩�A�* 

Average reward per step�=���Gm�       ��2	�]'�˩�A�*

epsilon�=����xp.       ��W�	�(�˩�A�* 

Average reward per step�=����       ��2	̘(�˩�A�*

epsilon�=����.       ��W�	`�*�˩�A�* 

Average reward per step�=�����       ��2	��*�˩�A�*

epsilon�=��;Q��.       ��W�	�s,�˩�A�* 

Average reward per step�=��Bۊ�       ��2	mt,�˩�A�*

epsilon�=��>�]m.       ��W�	W].�˩�A�* 

Average reward per step�=���S       ��2	�].�˩�A�*

epsilon�=��ցP*.       ��W�	��/�˩�A�* 

Average reward per step�=��N�-       ��2	Z�/�˩�A�*

epsilon�=��K���.       ��W�	A1�˩�A�* 

Average reward per step�=���F�}       ��2	�1�˩�A�*

epsilon�=�����.       ��W�	ff3�˩�A�* 

Average reward per step�=��2�{�       ��2	8g3�˩�A�*

epsilon�=��#��.       ��W�	ס4�˩�A�* 

Average reward per step�=��$r~�       ��2	�4�˩�A�*

epsilon�=��ʾi`.       ��W�	�}6�˩�A�* 

Average reward per step�=��~�       ��2	�~6�˩�A�*

epsilon�=��o#�J.       ��W�	y8�˩�A�* 

Average reward per step�=��K.6�       ��2	z8�˩�A�*

epsilon�=��!���.       ��W�	 |:�˩�A�* 

Average reward per step�=��/�h       ��2	�|:�˩�A�*

epsilon�=������.       ��W�	��;�˩�A�* 

Average reward per step�=����T       ��2	(�;�˩�A�*

epsilon�=��{h��.       ��W�	��=�˩�A�* 

Average reward per step�=���')�       ��2	p�=�˩�A�*

epsilon�=��&\�.       ��W�	s?�˩�A�* 

Average reward per step�=���7�       ��2	I�?�˩�A�*

epsilon�=���|�.       ��W�	z�@�˩�A�* 

Average reward per step�=��ҵ�       ��2	�@�˩�A�*

epsilon�=���y�D.       ��W�	��B�˩�A�* 

Average reward per step�=����ۦ       ��2	4�B�˩�A�*

epsilon�=���m�5.       ��W�	'�C�˩�A�* 

Average reward per step�=���8_�       ��2	��C�˩�A�*

epsilon�=��nus�.       ��W�	U�E�˩�A�* 

Average reward per step�=�����       ��2	�E�˩�A�*

epsilon�=���L��.       ��W�	��G�˩�A�* 

Average reward per step�=���bu�       ��2	h�G�˩�A�*

epsilon�=��J�c.       ��W�	��I�˩�A�* 

Average reward per step�=������       ��2	<�I�˩�A�*

epsilon�=�����.       ��W�	�K�˩�A�* 

Average reward per step�=����|       ��2	oK�˩�A�*

epsilon�=�����".       ��W�	��L�˩�A�* 

Average reward per step�=���K!A       ��2	[�L�˩�A�*

epsilon�=�����.       ��W�	_%N�˩�A�* 

Average reward per step�=���*�       ��2	>&N�˩�A�*

epsilon�=���ǅ�.       ��W�	�	P�˩�A�* 

Average reward per step�=��hf�       ��2	�
P�˩�A�*

epsilon�=���I	.       ��W�	5AQ�˩�A�* 

Average reward per step�=��G�K       ��2	�AQ�˩�A�*

epsilon�=���Z��.       ��W�	:S�˩�A�* 

Average reward per step�=��+�B�       ��2	2 S�˩�A�*

epsilon�=��^?�.       ��W�	��T�˩�A�* 

Average reward per step�=��r���       ��2	;�T�˩�A�*

epsilon�=���.       ��W�	b/V�˩�A�* 

Average reward per step�=��k�       ��2	+0V�˩�A�*

epsilon�=���ި�.       ��W�	3X�˩�A�* 

Average reward per step�=��"��       ��2	�3X�˩�A�*

epsilon�=����qx0       ���_	;QX�˩�A*#
!
Average reward per episode  z��5�.       ��W�	�QX�˩�A*!

total reward per episode  ��~��O.       ��W�	:u[�˩�A�* 

Average reward per step  z�h��       ��2	�u[�˩�A�*

epsilon  z����.       ��W�	�r]�˩�A�* 

Average reward per step  z��\�       ��2	ys]�˩�A�*

epsilon  z��Y�h.       ��W�	nk_�˩�A�* 

Average reward per step  z���P       ��2	Dl_�˩�A�*

epsilon  z�����.       ��W�	�ea�˩�A�* 

Average reward per step  z�Å[m       ��2	sfa�˩�A�*

epsilon  z�pK�.       ��W�	�Rc�˩�A�* 

Average reward per step  z�>��       ��2	XSc�˩�A�*

epsilon  z���h.       ��W�	+Le�˩�A�* 

Average reward per step  z��R��       ��2	�Le�˩�A�*

epsilon  z�gO	�.       ��W�	��f�˩�A�* 

Average reward per step  z��bV�       ��2	]�f�˩�A�*

epsilon  z�:f
�.       ��W�	�lh�˩�A�* 

Average reward per step  z�=�;�       ��2	�mh�˩�A�*

epsilon  z�f��.       ��W�	ͯi�˩�A�* 

Average reward per step  z��Y�X       ��2	h�i�˩�A�*

epsilon  z�_�3�.       ��W�	(�k�˩�A�* 

Average reward per step  z�A(�?       ��2	�k�˩�A�*

epsilon  z���.       ��W�	�zm�˩�A�* 

Average reward per step  z�/�c       ��2	J{m�˩�A�*

epsilon  z����.       ��W�	Dko�˩�A�* 

Average reward per step  z��?9       ��2	�ko�˩�A�*

epsilon  z�)�V.       ��W�	�p�˩�A�* 

Average reward per step  z���k�       ��2	ܡp�˩�A�*

epsilon  z����.       ��W�	Ƌr�˩�A�* 

Average reward per step  z�rnU       ��2	��r�˩�A�*

epsilon  z��|8Q.       ��W�	�ut�˩�A�* 

Average reward per step  z��Mڠ       ��2	�vt�˩�A�*

epsilon  z��"��.       ��W�	ٱu�˩�A�* 

Average reward per step  z����       ��2	��u�˩�A�*

epsilon  z��-��.       ��W�	��w�˩�A�* 

Average reward per step  z��ܯ�       ��2	-�w�˩�A�*

epsilon  z��Uh�.       ��W�	��x�˩�A�* 

Average reward per step  z�΁L�       ��2	K�x�˩�A�*

epsilon  z�k#��.       ��W�	%�z�˩�A�* 

Average reward per step  z��<��       ��2	��z�˩�A�*

epsilon  z�Ѻy.       ��W�	�|�˩�A�* 

Average reward per step  z�����       ��2	��|�˩�A�*

epsilon  z�G\��.       ��W�	i�}�˩�A�* 

Average reward per step  z�̃I8       ��2	��}�˩�A�*

epsilon  z�\V;�.       ��W�	e��˩�A�* 

Average reward per step  z�H�_       ��2	6��˩�A�*

epsilon  z���_&.       ��W�	��˩�A�* 

Average reward per step  z�lT       ��2	괁�˩�A�*

epsilon  z���.       ��W�	����˩�A�* 

Average reward per step  z���@#       ��2	M���˩�A�*

epsilon  z�[ԛ].       ��W�	���˩�A�* 

Average reward per step  z���b|       ��2	���˩�A�*

epsilon  z�J�J�.       ��W�	�Ά�˩�A�* 

Average reward per step  z�f�>*       ��2	І�˩�A�*

epsilon  z�u)�).       ��W�	���˩�A�* 

Average reward per step  z��QC�       ��2	Ҭ��˩�A�*

epsilon  z�o��@.       ��W�	��˩�A�* 

Average reward per step  z�-�&       ��2	���˩�A�*

epsilon  z�j�?�.       ��W�	ڋ�˩�A�* 

Average reward per step  z�j2�       ��2	/ۋ�˩�A�*

epsilon  z���.       ��W�	hЍ�˩�A�* 

Average reward per step  z�-]�       ��2	$э�˩�A�*

epsilon  z��T:.       ��W�	,��˩�A�* 

Average reward per step  z�_;8�       ��2	I��˩�A�*

epsilon  z���.       ��W�	���˩�A�* 

Average reward per step  z���&M       ��2	���˩�A�*

epsilon  z�fK�[.       ��W�	{ْ�˩�A�* 

Average reward per step  z�-�֪       ��2	@ڒ�˩�A�*

epsilon  z�6@@.       ��W�	�'��˩�A�* 

Average reward per step  z�=�5�       ��2	V(��˩�A�*

epsilon  z�+U�x.       ��W�	���˩�A�* 

Average reward per step  z���)]       ��2	���˩�A�*

epsilon  z���B0       ���_	*��˩�A*#
!
Average reward per episode|�W�pЩR.       ��W�	�*��˩�A*!

total reward per episode  ��W� �.       ��W�	���˩�A�* 

Average reward per step|�W�iP'U       ��2	k��˩�A�*

epsilon|�W�`S��.       ��W�	�+��˩�A�* 

Average reward per step|�W���0       ��2	g,��˩�A�*

epsilon|�W�H\d.       ��W�	���˩�A�* 

Average reward per step|�W�;��0       ��2	l��˩�A�*

epsilon|�W�)���.       ��W�	�D��˩�A�* 

Average reward per step|�W��9��       ��2	ZE��˩�A�*

epsilon|�W��h/�.       ��W�	9*��˩�A�* 

Average reward per step|�W�����       ��2	�*��˩�A�*

epsilon|�W�>��.       ��W�	��˩�A�* 

Average reward per step|�W��� �       ��2	���˩�A�*

epsilon|�W�j E�.       ��W�	�D��˩�A�* 

Average reward per step|�W��ٞ       ��2	E��˩�A�*

epsilon|�W����.       ��W�	���˩�A�* 

Average reward per step|�W��/       ��2	v��˩�A�*

epsilon|�W�V�,D.       ��W�	����˩�A�* 

Average reward per step|�W�����       ��2	h���˩�A�*

epsilon|�W��t��.       ��W�	V���˩�A�* 

Average reward per step|�W�^���       ��2	���˩�A�*

epsilon|�W�-#��.       ��W�	Hj��˩�A�* 

Average reward per step|�W�=�F�       ��2	�j��˩�A�*

epsilon|�W�*'q.       ��W�	�e��˩�A�* 

Average reward per step|�W�)a��       ��2	Vf��˩�A�*

epsilon|�W�7�Jl.       ��W�	&U��˩�A�* 

Average reward per step|�W��?��       ��2	*V��˩�A�*

epsilon|�W���!.       ��W�	����˩�A�* 

Average reward per step|�W��J��       ��2	B���˩�A�*

epsilon|�W�ѷ�.       ��W�	�z��˩�A�* 

Average reward per step|�W�P3�.       ��2	�{��˩�A�*

epsilon|�W�LQu�.       ��W�	ގ��˩�A�* 

Average reward per step|�W��V/�       ��2	u���˩�A�*

epsilon|�W��(<.       ��W�	�x��˩�A�* 

Average reward per step|�W�!@��       ��2	�y��˩�A�*

epsilon|�W����.       ��W�	f��˩�A�* 

Average reward per step|�W���;�       ��2	�f��˩�A�*

epsilon|�W�yftX.       ��W�	X���˩�A�* 

Average reward per step|�W��..i       ��2	���˩�A�*

epsilon|�W��A� .       ��W�	&���˩�A�* 

Average reward per step|�W�H�o�       ��2	Ύ��˩�A�*

epsilon|�W�5��i0       ���_	���˩�A*#
!
Average reward per episode33��7�.       ��W�	����˩�A*!

total reward per episode  )���.       ��W�	Ww��˩�A�* 

Average reward per step33�,ޢ       ��2	)x��˩�A�*

epsilon33�t.       ��W�	�ÿ�˩�A�* 

Average reward per step33�O9x       ��2	�Ŀ�˩�A�*

epsilon33�k�8.       ��W�	���˩�A�* 

Average reward per step33��
�       ��2	����˩�A�*

epsilon33�5ڬ�.       ��W�	����˩�A�* 

Average reward per step33�2�|�       ��2	����˩�A�*

epsilon33�-�k.       ��W�	n���˩�A�* 

Average reward per step33�$Bk�       ��2	&���˩�A�*

epsilon33��s.       ��W�	����˩�A�* 

Average reward per step33�{���       ��2	Q���˩�A�*

epsilon33��LH�.       ��W�	���˩�A�* 

Average reward per step33��:       ��2	����˩�A�*

epsilon33��>� .       ��W�	H���˩�A�* 

Average reward per step33�N�ø       ��2	���˩�A�*

epsilon33���!.       ��W�	^���˩�A�* 

Average reward per step33�,/��       ��2	0���˩�A�*

epsilon33�=��.       ��W�	����˩�A�* 

Average reward per step33��I͜       ��2	)���˩�A�*

epsilon33��2#�.       ��W�	�
��˩�A�* 

Average reward per step33�^���       ��2	k��˩�A�*

epsilon33�*}I.       ��W�	���˩�A�* 

Average reward per step33��=       ��2	����˩�A�*

epsilon33�f�ğ.       ��W�	@���˩�A�* 

Average reward per step33�����       ��2	����˩�A�*

epsilon33��# L.       ��W�	 ��˩�A�* 

Average reward per step33�bB��       ��2	� ��˩�A�*

epsilon33�H��.       ��W�	���˩�A�* 

Average reward per step33���u?       ��2	8��˩�A�*

epsilon33��E�.       ��W�	���˩�A�* 

Average reward per step33�๗       ��2	|��˩�A�*

epsilon33���0�.       ��W�	n���˩�A�* 

Average reward per step33�W��X       ��2		���˩�A�*

epsilon33�f(0�.       ��W�	\���˩�A�* 

Average reward per step33��z       ��2	����˩�A�*

epsilon33����.       ��W�	`��˩�A�* 

Average reward per step33����"       ��2	: ��˩�A�*

epsilon33��Q�R.       ��W�	���˩�A�* 

Average reward per step33�53S�       ��2	���˩�A�*

epsilon33���%�0       ���_	U2��˩�A*#
!
Average reward per episode�� ���.       ��W�	�2��˩�A*!

total reward per episode  !�O?.       ��W�	����˩�A�* 

Average reward per step�� ���^�       ��2	����˩�A�*

epsilon�� �T�pa.       ��W�	�G��˩�A�* 

Average reward per step�� ����       ��2	kH��˩�A�*

epsilon�� �߽$:.       ��W�	Z+��˩�A�* 

Average reward per step�� ��a       ��2	,��˩�A�*

epsilon�� �r壙.       ��W�	t��˩�A�* 

Average reward per step�� �|Qh�       ��2		��˩�A�*

epsilon�� �o�.       ��W�	5@��˩�A�* 

Average reward per step�� ��y�       ��2	A��˩�A�*

epsilon�� �I�R�.       ��W�	�&��˩�A�* 

Average reward per step�� ����       ��2	p'��˩�A�*

epsilon�� ��*�.       ��W�	��˩�A�* 

Average reward per step�� �ߤ�s       ��2	���˩�A�*

epsilon�� ��8��.       ��W�	|b��˩�A�* 

Average reward per step�� �W�l       ��2	c��˩�A�*

epsilon�� ��3�_.       ��W�	�Z��˩�A�* 

Average reward per step�� ���k�       ��2	�[��˩�A�*

epsilon�� �٢N.       ��W�	e���˩�A�* 

Average reward per step�� ����'       ��2	���˩�A�*

epsilon�� ���.       ��W�	G8��˩�A�* 

Average reward per step�� ��?       ��2	9��˩�A�*

epsilon�� ��k�
.       ��W�	����˩�A�* 

Average reward per step�� �zZԓ       ��2	a���˩�A�*

epsilon�� ���7:.       ��W�	mr��˩�A�* 

Average reward per step�� �����       ��2	s��˩�A�*

epsilon�� �����.       ��W�	�R��˩�A�* 

Average reward per step�� ���n�       ��2	TS��˩�A�*

epsilon�� ����.       ��W�	!���˩�A�* 

Average reward per step�� �>!�,       ��2	���˩�A�*

epsilon�� ��l~.       ��W�	ms��˩�A�* 

Average reward per step�� ��<�       ��2	Gt��˩�A�*

epsilon�� ����j.       ��W�	N^��˩�A�* 

Average reward per step�� ��c�       ��2	_��˩�A�*

epsilon�� �Da".       ��W�	�� �˩�A�* 

Average reward per step�� �h�       ��2	&� �˩�A�*

epsilon�� �I�.       ��W�	e��˩�A�* 

Average reward per step�� ��ݘ=       ��2	���˩�A�*

epsilon�� �v*�m.       ��W�	vp�˩�A�* 

Average reward per step�� ��/ۑ       ��2	Tq�˩�A�*

epsilon�� ��CN.       ��W�	��˩�A�* 

Average reward per step�� ��n,       ��2	ۿ�˩�A�*

epsilon�� �pu�.       ��W�	w��˩�A�* 

Average reward per step�� �ҬV<       ��2	
��˩�A�*

epsilon�� �&�>.       ��W�	p}	�˩�A�* 

Average reward per step�� �O���       ��2	~	�˩�A�*

epsilon�� �͖�d.       ��W�	��
�˩�A�* 

Average reward per step�� ��q�S       ��2	R�
�˩�A�*

epsilon�� ����@.       ��W�	��˩�A�* 

Average reward per step�� �zk6F       ��2	���˩�A�*

epsilon�� ���E.       ��W�	��˩�A�* 

Average reward per step�� �S��l       ��2	��˩�A�*

epsilon�� ��.       ��W�	���˩�A�* 

Average reward per step�� �.C��       ��2	 ��˩�A�*

epsilon�� ���1�.       ��W�	a��˩�A�* 

Average reward per step�� ���t�       ��2	.��˩�A�*

epsilon�� �䢖x.       ��W�	���˩�A�* 

Average reward per step�� �S��        ��2	v��˩�A�*

epsilon�� ��0ΰ.       ��W�	���˩�A�* 

Average reward per step�� ��'�       ��2	���˩�A�*

epsilon�� �]��.       ��W�	|��˩�A�* 

Average reward per step�� ���2       ��2	 ��˩�A�*

epsilon�� �_�,�.       ��W�	���˩�A�* 

Average reward per step�� ����       ��2	N��˩�A�*

epsilon�� ����.       ��W�	���˩�A�* 

Average reward per step�� �m���       ��2	J��˩�A�*

epsilon�� �S6�z.       ��W�	���˩�A�* 

Average reward per step�� �K�U@       ��2	B��˩�A�*

epsilon�� ���d�.       ��W�	��˩�A�* 

Average reward per step�� �"h�       ��2	���˩�A�*

epsilon�� �A���.       ��W�	˻�˩�A�* 

Average reward per step�� ��;��       ��2	b��˩�A�*

epsilon�� ��R'.       ��W�	�� �˩�A�* 

Average reward per step�� �J,Z�       ��2	�� �˩�A�*

epsilon�� �Iܮ.       ��W�	��"�˩�A�* 

Average reward per step�� ���W       ��2	��"�˩�A�*

epsilon�� ���/.       ��W�	�$�˩�A�* 

Average reward per step�� ���U#       ��2	�$�˩�A�*

epsilon�� �[X�.       ��W�	 &�˩�A�* 

Average reward per step�� ����       ��2	�&�˩�A�*

epsilon�� �Ү�.       ��W�	�'�˩�A�* 

Average reward per step�� �	rZ^       ��2	��'�˩�A�*

epsilon�� �1�~�.       ��W�	N%)�˩�A�* 

Average reward per step�� ����       ��2	�%)�˩�A�*

epsilon�� ���.       ��W�	��*�˩�A�* 

Average reward per step�� �L;.�       ��2	P +�˩�A�*

epsilon�� �Jx��.       ��W�	G�,�˩�A�* 

Average reward per step�� ��v       ��2	�,�˩�A�*

epsilon�� ��c�d.       ��W�	a5.�˩�A�* 

Average reward per step�� �9��u       ��2	&6.�˩�A�*

epsilon�� ��D_.       ��W�	Y0�˩�A�* 

Average reward per step�� ��(�-       ��2	�0�˩�A�*

epsilon�� �'�~ .       ��W�	�1�˩�A�* 

Average reward per step�� ���       ��2	��1�˩�A�*

epsilon�� ���O.       ��W�	�23�˩�A�* 

Average reward per step�� �֨w�       ��2	33�˩�A�*

epsilon�� �6Z�.       ��W�	B$5�˩�A�* 

Average reward per step�� ����       ��2	�$5�˩�A�*

epsilon�� �˷��0       ���_	E5�˩�A*#
!
Average reward per episode�S��p(.       ��W�	�E5�˩�A*!

total reward per episode  �¦Ktc.       ��W�	]9�˩�A�* 

Average reward per step�S�K��       ��2	&9�˩�A�*

epsilon�S�P�.       ��W�	XS:�˩�A�* 

Average reward per step�S�ȹL�       ��2	�S:�˩�A�*

epsilon�S�V�.       ��W�	�6<�˩�A�* 

Average reward per step�S�A�o�       ��2	v7<�˩�A�*

epsilon�S��~&v.       ��W�	�>�˩�A�* 

Average reward per step�S�Xh��       ��2	e>�˩�A�*

epsilon�S��A�.       ��W�	�W?�˩�A�* 

Average reward per step�S�u�]�       ��2	OX?�˩�A�*

epsilon�S�Y��W.       ��W�	AGA�˩�A�* 

Average reward per step�S���m       ��2	�GA�˩�A�*

epsilon�S��U,2.       ��W�	�"C�˩�A�* 

Average reward per step�S����       ��2	!#C�˩�A�*

epsilon�S�h�{.       ��W�	t]D�˩�A�* 

Average reward per step�S�iP��       ��2	^D�˩�A�*

epsilon�S�xZ�L.       ��W�	�?F�˩�A�* 

Average reward per step�S�E0�       ��2	�@F�˩�A�*

epsilon�S��z�V.       ��W�	�xG�˩�A�* 

Average reward per step�S�+3�       ��2	:yG�˩�A�*

epsilon�S�L�`.       ��W�	F`I�˩�A�* 

Average reward per step�S�Z.�       ��2	aI�˩�A�*

epsilon�S�F3D.       ��W�	JAK�˩�A�* 

Average reward per step�S��g       ��2	BK�˩�A�*

epsilon�S𿙦�7.       ��W�	R~L�˩�A�* 

Average reward per step�S𿳔ҝ       ��2	$L�˩�A�*

epsilon�S��Y�.       ��W�	�iN�˩�A�* 

Average reward per step�S��k{h       ��2	njN�˩�A�*

epsilon�S��,�.       ��W�	��O�˩�A�* 

Average reward per step�S�XF&y       ��2	�O�˩�A�*

epsilon�S�1u�.       ��W�	��Q�˩�A�* 

Average reward per step�S��Q       ��2	őQ�˩�A�*

epsilon�S𿒽��.       ��W�	�nS�˩�A�* 

Average reward per step�S�q-H�       ��2	/oS�˩�A�*

epsilon�S�W�.       ��W�	��T�˩�A�* 

Average reward per step�S�@KR�       ��2	��T�˩�A�*

epsilon�S�t�!�.       ��W�	E�V�˩�A�* 

Average reward per step�S𿑾>�       ��2	ؼV�˩�A�*

epsilon�S𿙆	�.       ��W�	�X�˩�A�* 

Average reward per step�S��g�       ��2	��X�˩�A�*

epsilon�S���/�.       ��W�	]�Z�˩�A�* 

Average reward per step�S��0S�       ��2	��Z�˩�A�*

epsilon�S�\� .       ��W�	-�[�˩�A�* 

Average reward per step�S�R��       ��2	�[�˩�A�*

epsilon�S��u(.       ��W�	W�]�˩�A�* 

Average reward per step�S���a       ��2	%�]�˩�A�*

epsilon�S���.       ��W�	;�_�˩�A�* 

Average reward per step�S��|l       ��2	Ύ_�˩�A�*

epsilon�S��K��.       ��W�	_�`�˩�A�* 

Average reward per step�S�aV5       ��2	)�`�˩�A�*

epsilon�S���t.       ��W�	�b�˩�A�* 

Average reward per step�S�����       ��2	�b�˩�A�*

epsilon�S�q��/.       ��W�	F�d�˩�A�* 

Average reward per step�S�%z/�       ��2	�d�˩�A�*

epsilon�S��Ҹ.       ��W�	E�e�˩�A�* 

Average reward per step�S��7�       ��2	 �e�˩�A�*

epsilon�S��S�F.       ��W�	�g�˩�A�* 

Average reward per step�S��:       ��2	��g�˩�A�*

epsilon�S�y D.       ��W�	7�i�˩�A�* 

Average reward per step�S�;��9       ��2	��i�˩�A�*

epsilon�S���v�.       ��W�	b�k�˩�A�* 

Average reward per step�S���
0       ��2	��k�˩�A�*

epsilon�S�k~��.       ��W�	�l�˩�A�* 

Average reward per step�S�r�       ��2	�l�˩�A�*

epsilon�S�<��.       ��W�	��n�˩�A�* 

Average reward per step�S�JJ,�       ��2	W�n�˩�A�*

epsilon�S��x	0       ���_	ko�˩�A	*#
!
Average reward per episode/�h�
���.       ��W�	o�˩�A	*!

total reward per episode  ��.H��.       ��W�	��r�˩�A�* 

Average reward per step/�h���       ��2	��r�˩�A�*

epsilon/�h��O�.       ��W�	��t�˩�A�* 

Average reward per step/�h� �MT       ��2	U�t�˩�A�*

epsilon/�h�7�N�.       ��W�	��v�˩�A�* 

Average reward per step/�h��}(6       ��2	��v�˩�A�*

epsilon/�h��Xst.       ��W�	�x�˩�A�* 

Average reward per step/�h�����       ��2	��x�˩�A�*

epsilon/�h�6vX.       ��W�	� {�˩�A�* 

Average reward per step/�h�д2�       ��2	�{�˩�A�*

epsilon/�h�z��.       ��W�	#�|�˩�A�* 

Average reward per step/�h�c�u       ��2	��|�˩�A�*

epsilon/�h���;.       ��W�	�/~�˩�A�* 

Average reward per step/�h�xt@�       ��2	n0~�˩�A�*

epsilon/�h���.       ��W�	f��˩�A�* 

Average reward per step/�h��ַ]       ��2	0��˩�A�*

epsilon/�h�0�P\.       ��W�	���˩�A�* 

Average reward per step/�h���Y�       ��2	���˩�A�*

epsilon/�h���T.       ��W�	�H��˩�A�* 

Average reward per step/�h�;��       ��2	4I��˩�A�*

epsilon/�h�4�]�.       ��W�	�6��˩�A�* 

Average reward per step/�h��[>       ��2	X7��˩�A�*

epsilon/�h�0"�U.       ��W�	2"��˩�A�* 

Average reward per step/�h�-MW       ��2	�"��˩�A�*

epsilon/�h���j.       ��W�	c��˩�A�* 

Average reward per step/�h�wGX       ��2	�c��˩�A�*

epsilon/�h�RҢ.       ��W�	:z��˩�A�* 

Average reward per step/�h�-� l       ��2	�z��˩�A�*

epsilon/�h�#U��.       ��W�	bi��˩�A�* 

Average reward per step/�h�V�       ��2	j��˩�A�*

epsilon/�h��4$.       ��W�	�`��˩�A�* 

Average reward per step/�h���\z       ��2	�a��˩�A�*

epsilon/�h����.       ��W�	T��˩�A�* 

Average reward per step/�h�~A�*       ��2	�T��˩�A�*

epsilon/�h�x@Yn.       ��W�	�,��˩�A�* 

Average reward per step/�h�=��V       ��2	�-��˩�A�*

epsilon/�h�l�9�.       ��W�	wf��˩�A�* 

Average reward per step/�h��UB       ��2	g��˩�A�*

epsilon/�h�I3�D.       ��W�	�?��˩�A�* 

Average reward per step/�h��龨       ��2	�@��˩�A�*

epsilon/�h���> .       ��W�	^���˩�A�* 

Average reward per step/�h�*^Q�       ��2	����˩�A�*

epsilon/�h��NK�.       ��W�	�h��˩�A�* 

Average reward per step/�h��{��       ��2	�i��˩�A�*

epsilon/�h��)%.       ��W�	hZ��˩�A�* 

Average reward per step/�h�.Ǉ�       ��2	�[��˩�A�*

epsilon/�h�昌b.       ��W�	����˩�A�* 

Average reward per step/�h�!�Sr       ��2	I���˩�A�*

epsilon/�h�JY��.       ��W�	����˩�A�* 

Average reward per step/�h�[�p�       ��2	����˩�A�*

epsilon/�h�b�?.       ��W�	�c��˩�A�* 

Average reward per step/�h��
;�       ��2	Ad��˩�A�*

epsilon/�h��\��.       ��W�	����˩�A�* 

Average reward per step/�h�R�-;       ��2	b���˩�A�*

epsilon/�h��[��.       ��W�	˝��˩�A�* 

Average reward per step/�h�_��@       ��2	����˩�A�*

epsilon/�h��5�.       ��W�	����˩�A�* 

Average reward per step/�h���       ��2	Y���˩�A�*

epsilon/�h����X.       ��W�	%ʥ�˩�A�* 

Average reward per step/�h��O��       ��2	�ʥ�˩�A�*

epsilon/�h���T�.       ��W�	���˩�A�* 

Average reward per step/�h���/       ��2	����˩�A�*

epsilon/�h�U4��.       ��W�	צ��˩�A�* 

Average reward per step/�h� ��       ��2	����˩�A�*

epsilon/�h�����0       ���_	�ũ�˩�A
*#
!
Average reward per episode  z���M�.       ��W�	�Ʃ�˩�A
*!

total reward per episode  ��̢&.       ��W�	����˩�A�* 

Average reward per step  z�JߵR       ��2	����˩�A�*

epsilon  z�
o
.       ��W�	V��˩�A�* 

Average reward per step  z�����       ��2	(��˩�A�*

epsilon  z�('Nh.       ��W�	u��˩�A�* 

Average reward per step  z�����       ��2	&��˩�A�*

epsilon  z�?�A.       ��W�		���˩�A�* 

Average reward per step  z��8��       ��2	����˩�A�*

epsilon  z��Ҝ�.       ��W�	¥��˩�A�* 

Average reward per step  z����y       ��2	����˩�A�*

epsilon  z�Z�q�.       ��W�	���˩�A�* 

Average reward per step  z�3Bd       ��2	e��˩�A�*

epsilon  z��y��.       ��W�	]ŷ�˩�A�* 

Average reward per step  z�����       ��2	.Ʒ�˩�A�*

epsilon  z�����.       ��W�	���˩�A�* 

Average reward per step  z�t�       ��2	[��˩�A�*

epsilon  z��.       ��W�	����˩�A�* 

Average reward per step  z�e�S�       ��2	p���˩�A�*

epsilon  z�~eG�.       ��W�	|��˩�A�* 

Average reward per step  z�#�       ��2	�|��˩�A�*

epsilon  z�_U*�.       ��W�	�l��˩�A�* 

Average reward per step  z��y<7       ��2	�m��˩�A�*

epsilon  z�����.       ��W�	qU��˩�A�* 

Average reward per step  z��Hfb       ��2	V��˩�A�*

epsilon  z�����.       ��W�	gH��˩�A�* 

Average reward per step  z�4e��       ��2	I��˩�A�*

epsilon  z���P.       ��W�	E���˩�A�* 

Average reward per step  z���Ŝ       ��2	���˩�A�*

epsilon  z�q���.       ��W�	j��˩�A�* 

Average reward per step  z���       ��2	#k��˩�A�*

epsilon  z�1{�.       ��W�	uW��˩�A�* 

Average reward per step  z��+.       ��2	KX��˩�A�*

epsilon  z�@�^�.       ��W�	����˩�A�* 

Average reward per step  z���       ��2	[���˩�A�*

epsilon  z��Y`.       ��W�	����˩�A�* 

Average reward per step  z�BapV       ��2	f���˩�A�*

epsilon  z��y�.       ��W�	�p��˩�A�* 

Average reward per step  z����        ��2	vq��˩�A�*

epsilon  z�e��.       ��W�	|���˩�A�* 

Average reward per step  z��Rn�       ��2	���˩�A�*

epsilon  z�R��C.       ��W�	����˩�A�* 

Average reward per step  z�/4�a       ��2	����˩�A�*

epsilon  z�ô�O.       ��W�	����˩�A�* 

Average reward per step  z�a~,�       ��2	����˩�A�*

epsilon  z����z.       ��W�	9'��˩�A�* 

Average reward per step  z���%       ��2	�'��˩�A�*

epsilon  z����.       ��W�	0��˩�A�* 

Average reward per step  z��@�       ��2	���˩�A�*

epsilon  z�\���.       ��W�	YQ��˩�A�* 

Average reward per step  z�h�]       ��2	�Q��˩�A�*

epsilon  z���0       ���_	"m��˩�A*#
!
Average reward per episode{��<+��.       ��W�	�m��˩�A*!

total reward per episode  � O��.       ��W�	�v��˩�A�* 

Average reward per step{����B       ��2	}w��˩�A�*

epsilon{���:�4.       ��W�	s��˩�A�* 

Average reward per step{��`;7�       ��2	�s��˩�A�*

epsilon{���*f.       ��W�	&W��˩�A�* 

Average reward per step{�����       ��2	:X��˩�A�*

epsilon{��F��.       ��W�	4d��˩�A�* 

Average reward per step{��Fg       ��2	�d��˩�A�*

epsilon{���s��.       ��W�	�V��˩�A�* 

Average reward per step{����.       ��2	uW��˩�A�*

epsilon{���S�X.       ��W�	><��˩�A�* 

Average reward per step{����.       ��2	=��˩�A�*

epsilon{���.       ��W�	�}��˩�A�* 

Average reward per step{��ߡ?       ��2	,~��˩�A�*

epsilon{�����.       ��W�	+n��˩�A�* 

Average reward per step{��Js��       ��2	�n��˩�A�*

epsilon{���uF.       ��W�	�[��˩�A�* 

Average reward per step{��a�<�       ��2	:\��˩�A�*

epsilon{��;);l.       ��W�	ٙ��˩�A�* 

Average reward per step{��/R�7       ��2	k���˩�A�*

epsilon{�����.       ��W�	t~��˩�A�* 

Average reward per step{��nʄ�       ��2	9��˩�A�*

epsilon{����Kv.       ��W�	�e��˩�A�* 

Average reward per step{���[w       ��2	ff��˩�A�*

epsilon{���sW�.       ��W�	ԝ��˩�A�* 

Average reward per step{����       ��2	s���˩�A�*

epsilon{����^.       ��W�	
���˩�A�* 

Average reward per step{���xV�       ��2	����˩�A�*

epsilon{��gM�g.       ��W�	g���˩�A�* 

Average reward per step{��L㱰       ��2	<���˩�A�*

epsilon{����U[.       ��W�	\���˩�A�* 

Average reward per step{����       ��2	)���˩�A�*

epsilon{��푈�.       ��W�	����˩�A�* 

Average reward per step{��BPm       ��2	����˩�A�*

epsilon{���%�J.       ��W�	����˩�A�* 

Average reward per step{��v ��       ��2	����˩�A�*

epsilon{��Z&'.       ��W�	{���˩�A�* 

Average reward per step{��9�-z       ��2	Q���˩�A�*

epsilon{��z�Bf.       ��W�	����˩�A�* 

Average reward per step{��$Zh       ��2	v���˩�A�*

epsilon{���f�.       ��W�	���˩�A�* 

Average reward per step{���O       ��2	����˩�A�*

epsilon{���%�.       ��W�	�� �˩�A�* 

Average reward per step{����?�       ��2	z� �˩�A�*

epsilon{��*��.       ��W�	���˩�A�* 

Average reward per step{��[�WL       ��2	���˩�A�*

epsilon{����׌.       ��W�	��˩�A�* 

Average reward per step{���H��       ��2	���˩�A�*

epsilon{���Q�.       ��W�	���˩�A�* 

Average reward per step{��;B��       ��2	���˩�A�*

epsilon{����k.       ��W�	�˩�A�* 

Average reward per step{��\��       ��2	��˩�A�*

epsilon{���[*.       ��W�	^�	�˩�A�* 

Average reward per step{���@R�       ��2	0�	�˩�A�*

epsilon{��s��k.       ��W�	���˩�A�* 

Average reward per step{���"       ��2	���˩�A�*

epsilon{���[�&.       ��W�	��˩�A�* 

Average reward per step{���NX�       ��2	;�˩�A�*

epsilon{��M�K�.       ��W�	[
�˩�A�* 

Average reward per step{������       ��2	$�˩�A�*

epsilon{��e��.       ��W�	���˩�A�* 

Average reward per step{��+	f�       ��2	���˩�A�*

epsilon{��&�.       ��W�	�4�˩�A�* 

Average reward per step{�����o       ��2	�5�˩�A�*

epsilon{��}�H*.       ��W�	�,�˩�A�* 

Average reward per step{��f�9       ��2	�-�˩�A�*

epsilon{���� �.       ��W�	Q.�˩�A�* 

Average reward per step{���Go       ��2	�.�˩�A�*

epsilon{���L��.       ��W�	��˩�A�* 

Average reward per step{�����       ��2	A�˩�A�*

epsilon{���w�.       ��W�	�N�˩�A�* 

Average reward per step{�����        ��2	"O�˩�A�*

epsilon{����F�.       ��W�	�4�˩�A�* 

Average reward per step{��d��       ��2	e5�˩�A�*

epsilon{�����!.       ��W�	"�˩�A�* 

Average reward per step{���Wc1       ��2	��˩�A�*

epsilon{����!!.       ��W�	si�˩�A�* 

Average reward per step{��j�m       ��2	8j�˩�A�*

epsilon{�� y.       ��W�	�U �˩�A�* 

Average reward per step{��O��H       ��2	�V �˩�A�*

epsilon{���Ժ.       ��W�	�G"�˩�A�* 

Average reward per step{������       ��2	�H"�˩�A�*

epsilon{��qQw�.       ��W�	V($�˩�A�* 

Average reward per step{��ɿ��       ��2	)$�˩�A�*

epsilon{��M���.       ��W�	l%�˩�A�* 

Average reward per step{��a�*�       ��2	�l%�˩�A�*

epsilon{���m�.       ��W�	�N'�˩�A�* 

Average reward per step{��W�;*       ��2	nO'�˩�A�*

epsilon{��v��L.       ��W�	v2)�˩�A�* 

Average reward per step{����/�       ��2	H3)�˩�A�*

epsilon{��/��
.       ��W�	�{*�˩�A�* 

Average reward per step{���<�       ��2	[|*�˩�A�*

epsilon{��(�.       ��W�	Ed,�˩�A�* 

Average reward per step{��`1}*       ��2	e,�˩�A�*

epsilon{���"�.       ��W�	�F.�˩�A�* 

Average reward per step{��i2΀       ��2	=G.�˩�A�*

epsilon{����`G0       ���_	�b.�˩�A*#
!
Average reward per episode  ��9��.       ��W�	^c.�˩�A*!

total reward per episode  ��t�.       ��W�	��1�˩�A�* 

Average reward per step  ��Dj"-       ��2	Q�1�˩�A�*

epsilon  ��� �h.       ��W�	�3�˩�A�* 

Average reward per step  �����@       ��2	��3�˩�A�*

epsilon  ����.       ��W�	�o5�˩�A�* 

Average reward per step  ��R��       ��2	�p5�˩�A�*

epsilon  �����.       ��W�	��6�˩�A�* 

Average reward per step  ����U�       ��2	7�6�˩�A�*

epsilon  ��$�� .       ��W�	m�8�˩�A�* 

Average reward per step  ������       ��2	C�8�˩�A�*

epsilon  ���?�.       ��W�	*�9�˩�A�* 

Average reward per step  ��%A%Q       ��2	��9�˩�A�*

epsilon  ���-t/.       ��W�	�;�˩�A�* 

Average reward per step  ��zK��       ��2	ޫ;�˩�A�*

epsilon  ��X�!..       ��W�	a�=�˩�A�* 

Average reward per step  ������       ��2	��=�˩�A�*

epsilon  ���A��.       ��W�	R�?�˩�A�* 

Average reward per step  �����8       ��2	�?�˩�A�*

epsilon  ���	.       ��W�	��@�˩�A�* 

Average reward per step  ����8�       ��2	��@�˩�A�*

epsilon  ���a�.       ��W�	��B�˩�A�* 

Average reward per step  ���H�       ��2	��B�˩�A�*

epsilon  ��<�=.       ��W�	��D�˩�A�* 

Average reward per step  ������       ��2	��D�˩�A�*

epsilon  ��ZX�G.       ��W�	�F�˩�A�* 

Average reward per step  ���n�       ��2	�F�˩�A�*

epsilon  ��BD=.       ��W�	8�G�˩�A�* 

Average reward per step  ���c��       ��2	��G�˩�A�*

epsilon  ��@_��.       ��W�	��I�˩�A�* 

Average reward per step  ��Xx2O       ��2	�I�˩�A�*

epsilon  ��'�N.       ��W�	�K�˩�A�* 

Average reward per step  ���$�^       ��2	��K�˩�A�*

epsilon  ����.       ��W�	�M�˩�A�* 

Average reward per step  �����6       ��2	aM�˩�A�*

epsilon  ��)��2.       ��W�	O�˩�A�* 

Average reward per step  �����       ��2	�O�˩�A�*

epsilon  ��K�.       ��W�	A�P�˩�A�* 

Average reward per step  ����       ��2	�P�˩�A�*

epsilon  ����1.       ��W�	��R�˩�A�* 

Average reward per step  ��(��       ��2	��R�˩�A�*

epsilon  �����&.       ��W�	��T�˩�A�* 

Average reward per step  ��^�rp       ��2	&�T�˩�A�*

epsilon  ��T�R .       ��W�	��V�˩�A�* 

Average reward per step  ����       ��2	1�V�˩�A�*

epsilon  ���6�.       ��W�	t	Y�˩�A�* 

Average reward per step  ����%       ��2	
Y�˩�A�*

epsilon  ����<.       ��W�	�[�˩�A�* 

Average reward per step  �����       ��2	7[�˩�A�*

epsilon  �����.       ��W�	a�\�˩�A�* 

Average reward per step  ���
�!       ��2	/�\�˩�A�*

epsilon  ���gr.       ��W�	%A^�˩�A�* 

Average reward per step  ����       ��2	�A^�˩�A�*

epsilon  ����͹.       ��W�	*`�˩�A�* 

Average reward per step  ���F�       ��2	�*`�˩�A�*

epsilon  ��T��.       ��W�	�b�˩�A�* 

Average reward per step  ����Z       ��2	�b�˩�A�*

epsilon  ��^�<.       ��W�	�Vc�˩�A�* 

Average reward per step  ��mu�       ��2	uWc�˩�A�*

epsilon  ���#
.       ��W�	_Ee�˩�A�* 

Average reward per step  ��|       ��2	|Fe�˩�A�*

epsilon  �����.       ��W�	�*g�˩�A�* 

Average reward per step  ����       ��2	0+g�˩�A�*

epsilon  ��Ij�$.       ��W�	�gh�˩�A�* 

Average reward per step  ����;       ��2	fhh�˩�A�*

epsilon  ���IQ�.       ��W�	+Lj�˩�A�* 

Average reward per step  ����O       ��2	�Lj�˩�A�*

epsilon  ����E.       ��W�	#/l�˩�A�* 

Average reward per step  ��[	�       ��2	0l�˩�A�*

epsilon  �����C.       ��W�	Djm�˩�A�* 

Average reward per step  �����       ��2	�jm�˩�A�*

epsilon  �����.       ��W�	*So�˩�A�* 

Average reward per step  ������       ��2	�So�˩�A�*

epsilon  ��� }.       ��W�	�p�˩�A�* 

Average reward per step  ��'��T       ��2	ڍp�˩�A�*

epsilon  ���7��.       ��W�	��r�˩�A�* 

Average reward per step  ������       ��2	�r�˩�A�*

epsilon  ��pZes.       ��W�	J`t�˩�A�* 

Average reward per step  ���d[2       ��2	,at�˩�A�*

epsilon  ����j�.       ��W�	B\v�˩�A�* 

Average reward per step  ��?|P       ��2	]v�˩�A�*

epsilon  ���x-�.       ��W�	��w�˩�A�* 

Average reward per step  ���X��       ��2	��w�˩�A�*

epsilon  ��.��.       ��W�	�y�˩�A�* 

Average reward per step  ���g#U       ��2	χy�˩�A�*

epsilon  ��JL?o.       ��W�	�|�˩�A�* 

Average reward per step  �����v       ��2	�|�˩�A�*

epsilon  ��\ /�.       ��W�	��}�˩�A�* 

Average reward per step  �����       ��2	C ~�˩�A�*

epsilon  ��=��.       ��W�	$��˩�A�* 

Average reward per step  ��n%T       ��2	���˩�A�*

epsilon  ���Y��.       ��W�	(��˩�A�* 

Average reward per step  ����l�       ��2	���˩�A�*

epsilon  ���`�".       ��W�	P7��˩�A�* 

Average reward per step  �����k       ��2	.8��˩�A�*

epsilon  ���CJ�.       ��W�	�J��˩�A�* 

Average reward per step  ����B       ��2	oK��˩�A�*

epsilon  ���q�.       ��W�	����˩�A�* 

Average reward per step  ��:��       ��2	B���˩�A�*

epsilon  ����V.       ��W�	[
��˩�A�* 

Average reward per step  ��f�yy       ��2	-��˩�A�*

epsilon  ��P�+�.       ��W�	����˩�A�* 

Average reward per step  ���,�       ��2	U���˩�A�*

epsilon  ��}��.       ��W�	Dތ�˩�A�* 

Average reward per step  ���K       ��2	ߌ�˩�A�*

epsilon  ����u.       ��W�	�2��˩�A�* 

Average reward per step  ��్�       ��2	n3��˩�A�*

epsilon  �����.       ��W�	��˩�A�* 

Average reward per step  ����       ��2	���˩�A�*

epsilon  ����]�.       ��W�	b���˩�A�* 

Average reward per step  ��;3�       ��2	����˩�A�*

epsilon  ��gs��.       ��W�	�5��˩�A�* 

Average reward per step  �����       ��2	�6��˩�A�*

epsilon  ��Z.       ��W�	1��˩�A�* 

Average reward per step  ������       ��2	�1��˩�A�*

epsilon  ��?ܻ7.       ��W�	"��˩�A�* 

Average reward per step  ��c���       ��2	���˩�A�*

epsilon  ��H\�.       ��W�	�X��˩�A�* 

Average reward per step  ����i       ��2	yY��˩�A�*

epsilon  ������.       ��W�	2=��˩�A�* 

Average reward per step  �����       ��2	�=��˩�A�*

epsilon  ���D�.       ��W�	�:��˩�A�* 

Average reward per step  ��       ��2	�;��˩�A�*

epsilon  ��(��.       ��W�	�J��˩�A�* 

Average reward per step  ���LA       ��2	�K��˩�A�*

epsilon  ����I�.       ��W�	�5��˩�A�* 

Average reward per step  ���E�       ��2	~6��˩�A�*

epsilon  ��S�Z�.       ��W�	���˩�A�* 

Average reward per step  ��2���       ��2	���˩�A�*

epsilon  ���C�.       ��W�	ff��˩�A�* 

Average reward per step  ����k�       ��2	g��˩�A�*

epsilon  ����I�.       ��W�	�e��˩�A�* 

Average reward per step  ��� ]       ��2	�f��˩�A�*

epsilon  �����7.       ��W�	U��˩�A�* 

Average reward per step  ���,y�       ��2	�U��˩�A�*

epsilon  ��d/��.       ��W�	2��˩�A�* 

Average reward per step  �����5       ��2	�2��˩�A�*

epsilon  ���Pe&.       ��W�	�t��˩�A�* 

Average reward per step  ������       ��2	�u��˩�A�*

epsilon  ���r�.       ��W�	�X��˩�A�* 

Average reward per step  ��ݽ�       ��2	\Y��˩�A�*

epsilon  ��E.\.       ��W�	К��˩�A�* 

Average reward per step  �����       ��2	c���˩�A�*

epsilon  �����.       ��W�	�˩�A�* 

Average reward per step  ���&�C       ��2	����˩�A�*

epsilon  ��2D�.       ��W�	���˩�A�* 

Average reward per step  ���ټ1       ��2	����˩�A�*

epsilon  ��MR=.       ��W�	����˩�A�* 

Average reward per step  ��O�       ��2	����˩�A�*

epsilon  ����;.       ��W�	Sx��˩�A�* 

Average reward per step  ��S 
�       ��2	 y��˩�A�*

epsilon  ����40       ���_	9���˩�A*#
!
Average reward per episodej���G.       ��W�	���˩�A*!

total reward per episode  8��H&�.       ��W�	�i��˩�A�* 

Average reward per stepj���O       ��2	jj��˩�A�*

epsilonj���d�.       ��W�	����˩�A�* 

Average reward per stepj�A:+1       ��2	C���˩�A�*

epsilonj���.       ��W�	����˩�A�* 

Average reward per stepj�g'&�       ��2	G���˩�A�*

epsilonj�����.       ��W�	휾�˩�A�* 

Average reward per stepj����       ��2	ǝ��˩�A�*

epsilonj�Aޙ�.       ��W�	E���˩�A�* 

Average reward per stepj�$F]6       ��2	ס��˩�A�*

epsilonj����.       ��W�	8���˩�A�* 

Average reward per stepj��k�       ��2	����˩�A�*

epsilonj��Ғb.       ��W�	���˩�A�* 

Average reward per stepj�x���       ��2	����˩�A�*

epsilonj�Y�).       ��W�	����˩�A�* 

Average reward per stepj�71i+       ��2	A���˩�A�*

epsilonj����q.       ��W�	;���˩�A�* 

Average reward per stepj�a�HR       ��2	����˩�A�*

epsilonj���X.       ��W�	����˩�A�* 

Average reward per stepj�Y(`?       ��2	����˩�A�*

epsilonj���YB.       ��W�	����˩�A�* 

Average reward per stepj�2K�       ��2	D���˩�A�*

epsilonj��+R
.       ��W�	q ��˩�A�* 

Average reward per stepj�%���       ��2	 ��˩�A�*

epsilonj�%U]�.       ��W�	���˩�A�* 

Average reward per stepj�"��       ��2	����˩�A�*

epsilonj����.       ��W�	���˩�A�* 

Average reward per stepj�Z<�       ��2	����˩�A�*

epsilonj�)(..       ��W�	c��˩�A�* 

Average reward per stepj�P{oN       ��2	=��˩�A�*

epsilonj��?�.       ��W�	���˩�A�* 

Average reward per stepj���C�       ��2	����˩�A�*

epsilonj��+
�0       ���_	h��˩�A*#
!
Average reward per episode  �d{x.       ��W�	���˩�A*!

total reward per episode  �`e*u.       ��W�	���˩�A�* 

Average reward per step  �)��       ��2	����˩�A�*

epsilon  ��,k.       ��W�	=��˩�A�* 

Average reward per step  ���(�       ��2	�=��˩�A�*

epsilon  ��0K.       ��W�	�7��˩�A�* 

Average reward per step  �	&       ��2	78��˩�A�*

epsilon  �|L�.       ��W�	y#��˩�A�* 

Average reward per step  �pK       ��2	p$��˩�A�*

epsilon  �I7�.       ��W�	��˩�A�* 

Average reward per step  �h���       ��2	���˩�A�*

epsilon  �Gm#.       ��W�	p���˩�A�* 

Average reward per step  ���        ��2	B���˩�A�*

epsilon  ���5.       ��W�	�,��˩�A�* 

Average reward per step  ��i�/       ��2	�-��˩�A�*

epsilon  �[R~u.       ��W�	/��˩�A�* 

Average reward per step  ��˫z       ��2	0��˩�A�*

epsilon  ���.       ��W�	%!��˩�A�* 

Average reward per step  ���       ��2	�!��˩�A�*

epsilon  ��U.       ��W�	���˩�A�* 

Average reward per step  ��9]       ��2	���˩�A�*

epsilon  �z:�.       ��W�	_\��˩�A�* 

Average reward per step  �CW�       ��2	%]��˩�A�*

epsilon  �@p��.       ��W�	S>��˩�A�* 

Average reward per step  �ގ��       ��2	1?��˩�A�*

epsilon  ����.       ��W�	#��˩�A�* 

Average reward per step  �����       ��2	�#��˩�A�*

epsilon  ���'.       ��W�	�a��˩�A�* 

Average reward per step  �H`       ��2	�b��˩�A�*

epsilon  �@S��.       ��W�	VF��˩�A�* 

Average reward per step  ��Ʋ�       ��2	$G��˩�A�*

epsilon  ��N+�.       ��W�	c+��˩�A�* 

Average reward per step  �Dru�       ��2	�+��˩�A�*

epsilon  ���"�.       ��W�	�p��˩�A�* 

Average reward per step  �
�,       ��2	7q��˩�A�*

epsilon  ���\.       ��W�	3p��˩�A�* 

Average reward per step  �� a�       ��2	 q��˩�A�*

epsilon  ���R�.       ��W�	�P��˩�A�* 

Average reward per step  ��ݾo       ��2	@Q��˩�A�*

epsilon  ��G��.       ��W�	�2��˩�A�* 

Average reward per step  ��Ū�       ��2	j3��˩�A�*

epsilon  �b3|.       ��W�	Oy��˩�A�* 

Average reward per step  �&�)=       ��2	Jz��˩�A�*

epsilon  ����.       ��W�	g��˩�A�* 

Average reward per step  �Ly��       ��2	h��˩�A�*

epsilon  ���H.       ��W�	�u��˩�A�* 

Average reward per step  ���Ԉ       ��2	>v��˩�A�*

epsilon  �;���.       ��W�	?o��˩�A�* 

Average reward per step  �S14       ��2	�o��˩�A�*

epsilon  ����.       ��W�	�b�˩�A�* 

Average reward per step  ��I�o       ��2	�c�˩�A�*

epsilon  ��3��.       ��W�	ӣ�˩�A�* 

Average reward per step  ��7)       ��2	{��˩�A�*

epsilon  �I ߴ.       ��W�	P��˩�A�* 

Average reward per step  ��x|       ��2	��˩�A�*

epsilon  ���.       ��W�	hu�˩�A�* 

Average reward per step  �"���       ��2	Kv�˩�A�*

epsilon  �x=�.       ��W�	���˩�A�* 

Average reward per step  ��3fU       ��2	k��˩�A�*

epsilon  ���r.       ��W�	�	�˩�A�* 

Average reward per step  ��O�       ��2	��	�˩�A�*

epsilon  �̽��.       ��W�	j��˩�A�* 

Average reward per step  ��
       ��2	D��˩�A�*

epsilon  �ʂ:J.       ��W�	j��˩�A�* 

Average reward per s