       �K"	   ̩�Abrain.Event:2��ܤ��      �*	j�8̩�A"��
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
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
end_mask*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
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
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*
Tshape0*0
_output_shapes
:������������������*
T0
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
dense_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�~�=
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
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
_output_shapes
:	�*
T0
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
VariableV2*
shape:�*
shared_name *
dtype0*
_output_shapes	
:�*
	container 
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:�*
T0*
_class
loc:@dense_1/bias
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
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
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
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������d*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������d
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
dense_3/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��L>*
dtype0
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
dense_3/ConstConst*
_output_shapes
:2*
valueB2*    *
dtype0
x
dense_3/bias
VariableV2*
shape:2*
shared_name *
dtype0*
_output_shapes
:2*
	container 
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
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
validate_shape(*
_output_shapes

:2*
use_locking(*
T0*!
_class
loc:@dense_4/kernel
{
dense_4/kernel/readIdentitydense_4/kernel*
_output_shapes

:2*
T0*!
_class
loc:@dense_4/kernel
Z
dense_4/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
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
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:
q
dense_4/bias/readIdentitydense_4/bias*
_output_shapes
:*
T0*
_class
loc:@dense_4/bias
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
activation_4/IdentityIdentitydense_4/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_5/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
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
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_5/kernel
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
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
T0*
Index0*
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
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
end_mask*'
_output_shapes
:���������*
Index0*
T0*
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 
|
flatten_1_input_1Placeholder*+
_output_shapes
:���������* 
shape:���������*
dtype0
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
!flatten_1_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
flatten_1_1/strided_sliceStridedSliceflatten_1_1/Shapeflatten_1_1/strided_slice/stack!flatten_1_1/strided_slice/stack_1!flatten_1_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:
[
flatten_1_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
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
dense_1_1/kernel/readIdentitydense_1_1/kernel*
T0*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	�
^
dense_1_1/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    
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
dense_2_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?�ʽ
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
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
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
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(*
_output_shapes

:d2*
use_locking(
�
dense_3_1/kernel/readIdentitydense_3_1/kernel*
_output_shapes

:d2*
T0*#
_class
loc:@dense_3_1/kernel
\
dense_3_1/ConstConst*
valueB2*    *
dtype0*
_output_shapes
:2
z
dense_3_1/bias
VariableV2*
shape:2*
shared_name *
dtype0*
_output_shapes
:2*
	container 
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
dense_4_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�D�>
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
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
shared_name *
dtype0
�
dense_4_1/kernel/AssignAssigndense_4_1/kerneldense_4_1/random_uniform*
_output_shapes

:2*
use_locking(*
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(
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
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
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
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
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
lambda_1_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
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
IsVariableInitialized_8IsVariableInitializeddense_5/kernel*
_output_shapes
: *!
_class
loc:@dense_5/kernel*
dtype0
�
IsVariableInitialized_9IsVariableInitializeddense_5/bias*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedAdam/iterations*
_output_shapes
: *"
_class
loc:@Adam/iterations*
dtype0	
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
AssignAssigndense_1_1/kernelPlaceholder*
_output_shapes
:	�*
use_locking( *
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(
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
Assign_5Assigndense_3_1/biasPlaceholder_5*
validate_shape(*
_output_shapes
:2*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias
^
Placeholder_6Placeholder*
shape
:2*
dtype0*
_output_shapes

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
Placeholder_8Placeholder*
shape
:*
dtype0*
_output_shapes

:
�
Assign_8Assigndense_5_1/kernelPlaceholder_8*
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:*
shape:
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
SGD/iterations/readIdentitySGD/iterations*!
_class
loc:@SGD/iterations*
_output_shapes
: *
T0	
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
lambda_1_sample_weightsPlaceholder*
shape:���������*
dtype0*#
_output_shapes
:���������
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
)loss/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:���������
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
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
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
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
SGD_1/iterations/readIdentitySGD_1/iterations*
_output_shapes
: *
T0	*#
_class
loc:@SGD_1/iterations
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
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@SGD_1/momentum*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*#
_output_shapes
:���������*
T0
f
loss_1/lambda_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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

loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*
T0*#
_output_shapes
:���������*

Tidx0*
	keep_dims( 
�
loss_targetPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
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
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*#
_output_shapes
:���������*
T0
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
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
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
4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
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
IsVariableInitialized_26IsVariableInitializedSGD/lr*
dtype0*
_output_shapes
: *
_class
loc:@SGD/lr
�
IsVariableInitialized_27IsVariableInitializedSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
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
dtype0*
_output_shapes
: *
_class
loc:@SGD_1/decay
�
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"l��aF      ��w	�d:̩�AJ��
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
flatten_1/ConstConst*
_output_shapes
:*
valueB: *
dtype0
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
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*0
_output_shapes
:������������������*
T0*
Tshape0
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:	�*
	container *
shape:	�
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
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:�*
T0*
_class
loc:@dense_1/bias
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
activation_1/ReluReludense_1/BiasAdd*(
_output_shapes
:����������*
T0
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
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	�d
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
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:���������d*
T0*
data_formatNHWC
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
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
_output_shapes

:d2*
T0
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
VariableV2*
shape:2*
shared_name *
dtype0*
_output_shapes
:2*
	container 
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
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*'
_output_shapes
:���������2*
T0*
data_formatNHWC
\
activation_3/ReluReludense_3/BiasAdd*
T0*'
_output_shapes
:���������2
m
dense_4/random_uniform/shapeConst*
_output_shapes
:*
valueB"2      *
dtype0
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
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
T0*
_class
loc:@dense_4/bias*
validate_shape(*
_output_shapes
:
q
dense_4/bias/readIdentitydense_4/bias*
_output_shapes
:*
T0*
_class
loc:@dense_4/bias
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
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�m?
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
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
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
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
m
lambda_1/strided_slice/stackConst*
_output_shapes
:*
valueB"        *
dtype0
o
lambda_1/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
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
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*'
_output_shapes
:���������*

Tdim0*
T0
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
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*
T0*'
_output_shapes
:���������
o
lambda_1/strided_slice_2/stackConst*
_output_shapes
:*
valueB"       *
dtype0
q
 lambda_1/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB"        
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
:���������*
Index0*
T0
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
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
�
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
_output_shapes
: *
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
Adam/beta_1/readIdentityAdam/beta_1*
_class
loc:@Adam/beta_1*
_output_shapes
: *
T0
^
Adam/beta_2/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*
dtype0
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
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(
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
flatten_1_1/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
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
:*
Index0*
T0
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
flatten_1_1/stack/0Const*
_output_shapes
: *
valueB :
���������*
dtype0
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
_output_shapes
:*
T0*

axis *
N
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
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
dtype0*
_output_shapes
:	�d*
seed2Ӽ�*
seed���)*
T0
�
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
T0*
_output_shapes
: 
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
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	�d*
use_locking(*
T0
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
dense_2_1/bias/readIdentitydense_2_1/bias*!
_class
loc:@dense_2_1/bias*
_output_shapes
:d*
T0
�
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:���������d*
transpose_a( *
transpose_b( 
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
VariableV2*
shape
:d2*
shared_name *
dtype0*
_output_shapes

:d2*
	container 
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
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
_output_shapes
:2*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(
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
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*'
_output_shapes
:���������2*
T0*
data_formatNHWC
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
&dense_4_1/random_uniform/RandomUniformRandomUniformdense_4_1/random_uniform/shape*
_output_shapes

:2*
seed2ֹM*
seed���)*
T0*
dtype0
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
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
T0*
_output_shapes

:2
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
dense_4_1/kernel/readIdentitydense_4_1/kernel*
_output_shapes

:2*
T0*#
_class
loc:@dense_4_1/kernel
\
dense_4_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
z
dense_4_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
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
 lambda_1_1/strided_slice/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
�
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:���������*
T0*
Index0*
shrink_axis_mask
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
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*
T0*'
_output_shapes
:���������
q
 lambda_1_1/strided_slice_2/stackConst*
_output_shapes
:*
valueB"       *
dtype0
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
valueB"       *
dtype0*
_output_shapes
:
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
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0
�
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_output_shapes
: *
_class
loc:@dense_3/bias*
dtype0
�
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_4/kernel
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
Adam/decay*
dtype0*
_output_shapes
: *
_class
loc:@Adam/decay
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
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*
dtype0*
_output_shapes
: *#
_class
loc:@dense_5_1/kernel
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
Assign_5Assigndense_3_1/biasPlaceholder_5*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2*
use_locking( 
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
Placeholder_7Placeholder*
dtype0*
_output_shapes
:*
shape:
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
Assign_8Assigndense_5_1/kernelPlaceholder_8*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking( *
T0
V
Placeholder_9Placeholder*
_output_shapes
:*
shape:*
dtype0
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
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	
s
SGD/iterations/readIdentitySGD/iterations*!
_class
loc:@SGD/iterations*
_output_shapes
: *
T0	
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
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/lr
[
SGD/lr/readIdentitySGD/lr*
_output_shapes
: *
T0*
_class
loc:@SGD/lr
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
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
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
loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*#
_output_shapes
:���������*
T0
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
SGD_1/iterations/readIdentitySGD_1/iterations*
_output_shapes
: *
T0	*#
_class
loc:@SGD_1/iterations
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
SGD_1/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
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
SGD_1/momentum/AssignAssignSGD_1/momentumSGD_1/momentum/initial_value*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@SGD_1/momentum*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
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
lambda_1_target_1Placeholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
t
lambda_1_sample_weights_1Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
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
loss_targetPlaceholder*%
shape:������������������*
dtype0*0
_output_shapes
:������������������
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
'loss_3/loss_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
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
loss_3/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
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
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
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
#metrics_2/mean_absolute_error/ConstConst*
_output_shapes
:*
valueB: *
dtype0
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
IsVariableInitialized_31IsVariableInitializedSGD_1/momentum*
dtype0*
_output_shapes
: *!
_class
loc:@SGD_1/momentum
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
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0Ě�.       ��W�	Y ̩�A*#
!
Average reward per episode�$��t�J,       ���E	P ̩�A*!

total reward per episode  �D1ѻ-       <A��	��#̩�A* 

Average reward per step�$��+�Q       `/�#	P�#̩�A*

epsilon�$����&�-       <A��	�)%̩�A* 

Average reward per step�$����ta       `/�#	�*%̩�A*

epsilon�$���6e�-       <A��	�'̩�A* 

Average reward per step�$��hBY�       `/�#	�'̩�A*

epsilon�$���:b[-       <A��	ZF(̩�A* 

Average reward per step�$��0�U�       `/�#	G(̩�A*

epsilon�$��j��-       <A��	k(*̩�A * 

Average reward per step�$�����       `/�#	�(*̩�A *

epsilon�$���7�-       <A��	�J,̩�A!* 

Average reward per step�$�����       `/�#	�K,̩�A!*

epsilon�$��8��-       <A��	�+.̩�A"* 

Average reward per step�$�����       `/�#	w,.̩�A"*

epsilon�$����Z�-       <A��	�	0̩�A#* 

Average reward per step�$����n       `/�#	c
0̩�A#*

epsilon�$��2-       <A��	�>1̩�A$* 

Average reward per step�$���atD       `/�#	O?1̩�A$*

epsilon�$��DL�-       <A��	y3̩�A%* 

Average reward per step�$��S��       `/�#	3̩�A%*

epsilon�$���j*-       <A��	�Y4̩�A&* 

Average reward per step�$�����       `/�#	2Z4̩�A&*

epsilon�$��<�P-       <A��	�76̩�A'* 

Average reward per step�$��sOt|       `/�#	;86̩�A'*

epsilon�$��5a�V-       <A��	Um7̩�A(* 

Average reward per step�$��ߤ�	       `/�#	�m7̩�A(*

epsilon�$��dBTv-       <A��	U9̩�A)* 

Average reward per step�$��= �       `/�#	 V9̩�A)*

epsilon�$���@�D-       <A��	9D;̩�A** 

Average reward per step�$��P       `/�#	�D;̩�A**

epsilon�$��5�a�-       <A��	�Q=̩�A+* 

Average reward per step�$����L�       `/�#	�R=̩�A+*

epsilon�$��Ze۱-       <A��	�B?̩�A,* 

Average reward per step�$���S<       `/�#	5C?̩�A,*

epsilon�$��U�˨-       <A��	|}@̩�A-* 

Average reward per step�$���|�Q       `/�#	~@̩�A-*

epsilon�$��Hp��-       <A��	mWB̩�A.* 

Average reward per step�$��B��       `/�#	6XB̩�A.*

epsilon�$������-       <A��	e8D̩�A/* 

Average reward per step�$��B���       `/�#	.9D̩�A/*

epsilon�$����ư-       <A��	�sE̩�A0* 

Average reward per step�$�����       `/�#	tE̩�A0*

epsilon�$��?�Q-       <A��	h\G̩�A1* 

Average reward per step�$����       `/�#	)]G̩�A1*

epsilon�$���~͘-       <A��	�FI̩�A2* 

Average reward per step�$��Ff�       `/�#	cGI̩�A2*

epsilon�$����N�-       <A��	1~J̩�A3* 

Average reward per step�$�����$       `/�#	�~J̩�A3*

epsilon�$���� �-       <A��	gL̩�A4* 

Average reward per step�$��z�       `/�#	�gL̩�A4*

epsilon�$��]zӽ-       <A��	�TN̩�A5* 

Average reward per step�$����+�       `/�#	�UN̩�A5*

epsilon�$��s���-       <A��	��O̩�A6* 

Average reward per step�$���7��       `/�#	�O̩�A6*

epsilon�$��R�BE-       <A��	7lQ̩�A7* 

Average reward per step�$��Gb��       `/�#	�lQ̩�A7*

epsilon�$���9/-       <A��	�R̩�A8* 

Average reward per step�$��N��       `/�#	��R̩�A8*

epsilon�$��1̮-       <A��	��T̩�A9* 

Average reward per step�$���٣�       `/�#	R�T̩�A9*

epsilon�$���t�-       <A��	��V̩�A:* 

Average reward per step�$�����%       `/�#	8�V̩�A:*

epsilon�$��;0K-       <A��	%�X̩�A;* 

Average reward per step�$��g��1       `/�#	ɐX̩�A;*

epsilon�$���?��-       <A��	WwZ̩�A<* 

Average reward per step�$��e@�*       `/�#	�wZ̩�A<*

epsilon�$����0       ���_	ՒZ̩�A*#
!
Average reward per episode>x��<H.       ��W�	}�Z̩�A*!

total reward per episode   ���-       <A��	�]̩�A=* 

Average reward per step>x��t�       `/�#	��]̩�A=*

epsilon>x����-       <A��	+�_̩�A>* 

Average reward per step>x�����       `/�#	ƣ_̩�A>*

epsilon>x����X-       <A��	d�a̩�A?* 

Average reward per step>x�T�?	       `/�#	9�a̩�A?*

epsilon>x��D�-       <A��	��b̩�A@* 

Average reward per step>x�{�˖       `/�#	��b̩�A@*

epsilon>x�����-       <A��	�d̩�AA* 

Average reward per step>x��gv�       `/�#	x�d̩�AA*

epsilon>x�ù��-       <A��	��e̩�AB* 

Average reward per step>x��悚       `/�#	}�e̩�AB*

epsilon>x��o�J-       <A��	��g̩�AC* 

Average reward per step>x�X��       `/�#	 �g̩�AC*

epsilon>x���*�-       <A��	�h̩�AD* 

Average reward per step>x����       `/�#	��h̩�AD*

epsilon>x�eG��-       <A��	��j̩�AE* 

Average reward per step>x�?_�       `/�#	6�j̩�AE*

epsilon>x����%-       <A��	�l̩�AF* 

Average reward per step>x���o       `/�#	��l̩�AF*

epsilon>x�$�71-       <A��	��m̩�AG* 

Average reward per step>x��,]�       `/�#	3�m̩�AG*

epsilon>x��l3�-       <A��	��o̩�AH* 

Average reward per step>x��O�S       `/�#	 �o̩�AH*

epsilon>x�5�n�-       <A��	 q̩�AI* 

Average reward per step>x��0`       `/�#	�q̩�AI*

epsilon>x�Lhx�-       <A��	�r̩�AJ* 

Average reward per step>x��f��       `/�#	��r̩�AJ*

epsilon>x�̆.f-       <A��	��t̩�AK* 

Average reward per step>x�W��       `/�#	�t̩�AK*

epsilon>x��g�-       <A��	�u̩�AL* 

Average reward per step>x���       `/�#	��u̩�AL*

epsilon>x�OG�,-       <A��	��w̩�AM* 

Average reward per step>x�6
��       `/�#	��w̩�AM*

epsilon>x�雮�-       <A��	�y̩�AN* 

Average reward per step>x���8Q       `/�#	Iy̩�AN*

epsilon>x�SF�-       <A��	� {̩�AO* 

Average reward per step>x���       `/�#	�{̩�AO*

epsilon>x��Q�!-       <A��	��|̩�AP* 

Average reward per step>x��U G       `/�#	��|̩�AP*

epsilon>x�l�-       <A��	9~̩�AQ* 

Average reward per step>x����C       `/�#	�~̩�AQ*

epsilon>x�|r�-       <A��	�̩�AR* 

Average reward per step>x��F�0       `/�#	��̩�AR*

epsilon>x��]g-       <A��	�'�̩�AS* 

Average reward per step>x��D       `/�#	�(�̩�AS*

epsilon>x�pn�,-       <A��	��̩�AT* 

Average reward per step>x�5��       `/�#	Q�̩�AT*

epsilon>x���0       ���_	�-�̩�A*#
!
Average reward per episodeUU���_k.       ��W�	E.�̩�A*!

total reward per episode  �T�MB-       <A��	J�̩�AU* 

Average reward per stepUU���ö       `/�#	(��̩�AU*

epsilonUU��R��-       <A��	�,�̩�AV* 

Average reward per stepUU����9       `/�#	E-�̩�AV*

epsilonUU��E��-       <A��	{�̩�AW* 

Average reward per stepUU���J��       `/�#	<�̩�AW*

epsilonUU���8�-       <A��	�H�̩�AX* 

Average reward per stepUU��7f�       `/�#	RI�̩�AX*

epsilonUU��'�s8-       <A��	�6�̩�AY* 

Average reward per stepUU����F�       `/�#	m7�̩�AY*

epsilonUU���jh-       <A��	�,�̩�AZ* 

Average reward per stepUU���ť       `/�#	�-�̩�AZ*

epsilonUU��Ke��-       <A��	w�̩�A[* 

Average reward per stepUU��7z�       `/�#		�̩�A[*

epsilonUU����-       <A��	$a�̩�A\* 

Average reward per stepUU����x8       `/�#	�a�̩�A\*

epsilonUU���a�-       <A��	�~�̩�A]* 

Average reward per stepUU��R䖐       `/�#	��̩�A]*

epsilonUU��)'{@-       <A��	���̩�A^* 

Average reward per stepUU�����       `/�#	��̩�A^*

epsilonUU��Es�U-       <A��	�t�̩�A_* 

Average reward per stepUU����F       `/�#	yu�̩�A_*

epsilonUU��G�b-       <A��	�h�̩�A`* 

Average reward per stepUU��y��       `/�#	/i�̩�A`*

epsilonUU��<�w�-       <A��	�i�̩�Aa* 

Average reward per stepUU��Fm��       `/�#	]j�̩�Aa*

epsilonUU��Z�-       <A��		��̩�Ab* 

Average reward per stepUU��rښ�       `/�#	ڍ�̩�Ab*

epsilonUU����ْ-       <A��	}�̩�Ac* 

Average reward per stepUU��nr�S       `/�#	�}�̩�Ac*

epsilonUU��y;�-       <A��	<i�̩�Ad* 

Average reward per stepUU���hl�       `/�#	�i�̩�Ad*

epsilonUU���]P�0       ���_	3��̩�A*#
!
Average reward per episode  ���bg.       ��W�	߉�̩�A*!

total reward per episode  ��H*-       <A��	�e�̩�Ae* 

Average reward per step  ��M�E       `/�#	Qf�̩�Ae*

epsilon  ���-       <A��	�Ƨ̩�Af* 

Average reward per step  ��d�"       `/�#	eǧ̩�Af*

epsilon  �-WU�-       <A��	t��̩�Ag* 

Average reward per step  �Y�o"       `/�#	F��̩�Ag*

epsilon  �{�m0-       <A��	n��̩�Ah* 

Average reward per step  ���UR       `/�#	��̩�Ah*

epsilon  �e�Hh-       <A��	ǟ�̩�Ai* 

Average reward per step  �%�       `/�#	b��̩�Ai*

epsilon  �H��-       <A��	���̩�Aj* 

Average reward per step  �p�       `/�#	|��̩�Aj*

epsilon  �����-       <A��	��̩�Ak* 

Average reward per step  �q�]�       `/�#	���̩�Ak*

epsilon  �O̹-       <A��	?�̩�Al* 

Average reward per step  �4ɥ�       `/�#	�?�̩�Al*

epsilon  �t���-       <A��	Z��̩�Am* 

Average reward per step  �fMV�       `/�#	蠴̩�Am*

epsilon  ��]�y-       <A��	��̩�An* 

Average reward per step  ��"��       `/�#	���̩�An*

epsilon  ��*:u-       <A��	�ٷ̩�Ao* 

Average reward per step  �~�$       `/�#	ڷ̩�Ao*

epsilon  �_���-       <A��	}ι̩�Ap* 

Average reward per step  �|��       `/�#	FϹ̩�Ap*

epsilon  ���EE-       <A��	�ɻ̩�Aq* 

Average reward per step  ���j       `/�#	�ʻ̩�Aq*

epsilon  �u2�O-       <A��	e��̩�Ar* 

Average reward per step  �j<{�       `/�#	&��̩�Ar*

epsilon  �>���-       <A��	���̩�As* 

Average reward per step  ��       `/�#	��̩�As*

epsilon  ���-       <A��	c��̩�At* 

Average reward per step  ��;d       `/�#	,��̩�At*

epsilon  �xi��-       <A��	���̩�Au* 

Average reward per step  �Jf�&       `/�#	\ �̩�Au*

epsilon  ���(-       <A��	���̩�Av* 

Average reward per step  �Þ[       `/�#	���̩�Av*

epsilon  �tӏ-       <A��	���̩�Aw* 

Average reward per step  �v�:       `/�#	���̩�Aw*

epsilon  ���3X-       <A��	s��̩�Ax* 

Average reward per step  �)[Yb       `/�#	E��̩�Ax*

epsilon  �.�be-       <A��	�D�̩�Ay* 

Average reward per step  ���p>       `/�#	�E�̩�Ay*

epsilon  ���-       <A��	RF�̩�Az* 

Average reward per step  ��o�       `/�#	,G�̩�Az*

epsilon  ����-       <A��	�O�̩�A{* 

Average reward per step  �b��       `/�#	vP�̩�A{*

epsilon  ��f-       <A��	W�̩�A|* 

Average reward per step  �y�܍       `/�#	�W�̩�A|*

epsilon  ��1�-       <A��	`w�̩�A}* 

Average reward per step  ��n       `/�#	Wx�̩�A}*

epsilon  �L)�!-       <A��	���̩�A~* 

Average reward per step  ��p�       `/�#	���̩�A~*

epsilon  �TQ(-       <A��	���̩�A* 

Average reward per step  �CG�Y       `/�#	��̩�A*

epsilon  �Ȣ�.       ��W�	˻�̩�A�* 

Average reward per step  ���5       ��2	���̩�A�*

epsilon  ���^�.       ��W�	���̩�A�* 

Average reward per step  �ŲT�       ��2	��̩�A�*

epsilon  ��	.       ��W�	���̩�A�* 

Average reward per step  �R��       ��2	���̩�A�*

epsilon  �ޚ�.       ��W�	���̩�A�* 

Average reward per step  ���       ��2	 ��̩�A�*

epsilon  ��dq�.       ��W�	��̩�A�* 

Average reward per step  �fNʆ       ��2	��̩�A�*

epsilon  �;A.       ��W�	��̩�A�* 

Average reward per step  ����       ��2	}�̩�A�*

epsilon  �J�M(.       ��W�	 `�̩�A�* 

Average reward per step  �mv�       ��2	a�̩�A�*

epsilon  �3M��.       ��W�	�̩�A�* 

Average reward per step  �=�Y�       ��2	��̩�A�*

epsilon  �g�0�.       ��W�	��̩�A�* 

Average reward per step  �Y��F       ��2	��̩�A�*

epsilon  �o���.       ��W�	���̩�A�* 

Average reward per step  ���       ��2	` �̩�A�*

epsilon  ��c�d.       ��W�	���̩�A�* 

Average reward per step  �L&0�       ��2	���̩�A�*

epsilon  �����.       ��W�	���̩�A�* 

Average reward per step  ��ʥ       ��2	y��̩�A�*

epsilon  �x�7�.       ��W�	���̩�A�* 

Average reward per step  �/��       ��2	5��̩�A�*

epsilon  �*�f.       ��W�	-'�̩�A�* 

Average reward per step  �w�h       ��2	�'�̩�A�*

epsilon  ��}�I.       ��W�	c�̩�A�* 

Average reward per step  ��a�       ��2	��̩�A�*

epsilon  �x��E0       ���_	�%�̩�A*#
!
Average reward per episode�A���*0.       ��W�	1&�̩�A*!

total reward per episode  ��`%ѯ.       ��W�	Z�̩�A�* 

Average reward per step�A��2       ��2	�̩�A�*

epsilon�A�O�A�.       ��W�	��̩�A�* 

Average reward per step�A�6�
�       ��2	w�̩�A�*

epsilon�A���UQ.       ��W�	vT�̩�A�* 

Average reward per step�A�s.�       ��2	TU�̩�A�*

epsilon�A��_�.       ��W�	�\̩�A�* 

Average reward per step�A��G��       ��2	S]̩�A�*

epsilon�A�70�.       ��W�	��̩�A�* 

Average reward per step�A�5 �       ��2	<�̩�A�*

epsilon�A�F�]..       ��W�	̩�A�* 

Average reward per step�A��Gr       ��2	�̩�A�*

epsilon�A��ۖ.       ��W�	o�̩�A�* 

Average reward per step�A��X�       ��2	M�̩�A�*

epsilon�A�q�d.       ��W�	�V	̩�A�* 

Average reward per step�A����       ��2	�W	̩�A�*

epsilon�A�B��.       ��W�	�{̩�A�* 

Average reward per step�A�]�N\       ��2	�|̩�A�*

epsilon�A��8�.       ��W�	�*̩�A�* 

Average reward per step�A����q       ��2	�+̩�A�*

epsilon�A��G.       ��W�	�z̩�A�* 

Average reward per step�A��b�W       ��2	t{̩�A�*

epsilon�A�Tq�O.       ��W�	|�̩�A�* 

Average reward per step�A���w�       ��2	M�̩�A�*

epsilon�A���$.       ��W�	��̩�A�* 

Average reward per step�A�����       ��2	[�̩�A�*

epsilon�A�j���.       ��W�	,�̩�A�* 

Average reward per step�A���       ��2	k�̩�A�*

epsilon�A��6�.       ��W�	̩�A�* 

Average reward per step�A��\]       ��2	�̩�A�*

epsilon�A��|%Q.       ��W�	��̩�A�* 

Average reward per step�A�����       ��2	��̩�A�*

epsilon�A��0�.       ��W�	�̩�A�* 

Average reward per step�A��h��       ��2	�̩�A�*

epsilon�A��P(g.       ��W�	�@̩�A�* 

Average reward per step�A�ArT=       ��2	�A̩�A�*

epsilon�A�e-�7.       ��W�	�6̩�A�* 

Average reward per step�A�$��       ��2	z7̩�A�*

epsilon�A�u��j0       ���_	)^̩�A*#
!
Average reward per episode����)��.       ��W�	_̩�A*!

total reward per episode  �`Ġ�.       ��W�	/"̩�A�* 

Average reward per step�����       ��2	�/"̩�A�*

epsilon����)�.       ��W�	�,$̩�A�* 

Average reward per step�����ޮ�       ��2	V-$̩�A�*

epsilon�������.       ��W�	[(&̩�A�* 

Average reward per step�������       ��2	)&̩�A�*

epsilon����6�&.       ��W�	q9(̩�A�* 

Average reward per step������L       ��2	:(̩�A�*

epsilon����[��	.       ��W�	@2*̩�A�* 

Average reward per step������       ��2	�2*̩�A�*

epsilon�����.       ��W�	",̩�A�* 

Average reward per step�����Q0       ��2	#,̩�A�*

epsilon�����6�j.       ��W�	.̩�A�* 

Average reward per step�������       ��2	\.̩�A�*

epsilon����b�.       ��W�	I-0̩�A�* 

Average reward per step����w��       ��2	�-0̩�A�*

epsilon�����g�Z.       ��W�	�(2̩�A�* 

Average reward per step�����^�#       ��2	�)2̩�A�*

epsilon����i: 6.       ��W�	�4̩�A�* 

Average reward per step�����<1�       ��2	�4̩�A�*

epsilon����=��s.       ��W�	�h5̩�A�* 

Average reward per step����2���       ��2	�i5̩�A�*

epsilon����f��.       ��W�	�Q7̩�A�* 

Average reward per step�����ѝ       ��2	�R7̩�A�*

epsilon����pi��.       ��W�	B9̩�A�* 

Average reward per step�����|�       ��2	�B9̩�A�*

epsilon����lK�.       ��W�	�~;̩�A�* 

Average reward per step�����QV       ��2	�;̩�A�*

epsilon�����%�.       ��W�	�=̩�A�* 

Average reward per step�����^%�       ��2	z�=̩�A�*

epsilon����y7".       ��W�	��?̩�A�* 

Average reward per step����YC{       ��2	��?̩�A�*

epsilon����G���.       ��W�	zA̩�A�* 

Average reward per step����?ȇ       ��2	�zA̩�A�*

epsilon����X�c.       ��W�	��C̩�A�* 

Average reward per step����$+       ��2	��C̩�A�*

epsilon������Z�.       ��W�	C�E̩�A�* 

Average reward per step�����x
       ��2	��E̩�A�*

epsilon����qyL.       ��W�	�G̩�A�* 

Average reward per step�����.E2       ��2	@�G̩�A�*

epsilon����(N.       ��W�	�oJ̩�A�* 

Average reward per step�����j�       ��2	apJ̩�A�*

epsilon����\�:�.       ��W�	C�K̩�A�* 

Average reward per step�����@h       ��2	�K̩�A�*

epsilon����� ^r.       ��W�	d�M̩�A�* 

Average reward per step����8K�c       ��2	�M̩�A�*

epsilon����m�[.       ��W�	,�P̩�A�* 

Average reward per step������L       ��2	�P̩�A�*

epsilon�����m�b.       ��W�	Y�Q̩�A�* 

Average reward per step����d-       ��2	*�Q̩�A�*

epsilon����u�`�.       ��W�	T̩�A�* 

Average reward per step����}bP�       ��2	�T̩�A�*

epsilon�����J��.       ��W�	B'V̩�A�* 

Average reward per step����C��       ��2	�'V̩�A�*

epsilon�����Ѣ�.       ��W�	�;X̩�A�* 

Average reward per step����R�JS       ��2	�<X̩�A�*

epsilon����0��\.       ��W�	�>Z̩�A�* 

Average reward per step����n�       ��2	)?Z̩�A�*

epsilon�����sQ%.       ��W�	6w\̩�A�* 

Average reward per step����F�       ��2	�w\̩�A�*

epsilon����q.       ��W�	Ll^̩�A�* 

Average reward per step�����qE�       ��2	m^̩�A�*

epsilon�����o�).       ��W�	D�`̩�A�* 

Average reward per step�����♰       ��2	�`̩�A�*

epsilon����vw�d.       ��W�	��b̩�A�* 

Average reward per step����'��k       ��2	{�b̩�A�*

epsilon����i��L.       ��W�	��d̩�A�* 

Average reward per step����Jꞧ       ��2	��d̩�A�*

epsilon�����'��.       ��W�	7�f̩�A�* 

Average reward per step����[���       ��2	�f̩�A�*

epsilon������^{0       ���_	��f̩�A*#
!
Average reward per episode�_�q�l�.       ��W�	t�f̩�A*!

total reward per episode  ����.       ��W�	��j̩�A�* 

Average reward per step�_����       ��2	��j̩�A�*

epsilon�_�Aڜ.       ��W�	�9l̩�A�* 

Average reward per step�_��~�       ��2	�:l̩�A�*

epsilon�_��+�V.       ��W�		�m̩�A�* 

Average reward per step�_�Y�s       ��2	��m̩�A�*

epsilon�_��U3�.       ��W�	� p̩�A�* 

Average reward per step�_�㲂+       ��2	qp̩�A�*

epsilon�_����.       ��W�	�r̩�A�* 

Average reward per step�_���I�       ��2	�r̩�A�*

epsilon�_�V�.       ��W�	0�s̩�A�* 

Average reward per step�_�57       ��2	�s̩�A�*

epsilon�_���GI.       ��W�	�@v̩�A�* 

Average reward per step�_�@.q~       ��2	�Av̩�A�*

epsilon�_���pc.       ��W�	��w̩�A�* 

Average reward per step�_��sWf       ��2	��w̩�A�*

epsilon�_�]�6.       ��W�	
�y̩�A�* 

Average reward per step�_��P��       ��2	��y̩�A�*

epsilon�_�+d�*.       ��W�	�{̩�A�* 

Average reward per step�_�r�c       ��2	��{̩�A�*

epsilon�_�x�.       ��W�	*~̩�A�* 

Average reward per step�_�AL�o       ��2	�~̩�A�*

epsilon�_�x6E.       ��W�	��̩�A�* 

Average reward per step�_����       ��2	��̩�A�*

epsilon�_�78�.       ��W�	��̩�A�* 

Average reward per step�_��3��       ��2	��̩�A�*

epsilon�_����.       ��W�	_@�̩�A�* 

Average reward per step�_�Sl�D       ��2	�@�̩�A�*

epsilon�_�c*|�.       ��W�	hʅ̩�A�* 

Average reward per step�_��&m       ��2	C˅̩�A�*

epsilon�_��oxc.       ��W�	
+�̩�A�* 

Average reward per step�_��u�       ��2	�+�̩�A�*

epsilon�_�xjd.       ��W�	D0�̩�A�* 

Average reward per step�_�㒘       ��2	1�̩�A�*

epsilon�_���f.       ��W�	�7�̩�A�* 

Average reward per step�_����+       ��2	�8�̩�A�*

epsilon�_�EpK.       ��W�	r�̩�A�* 

Average reward per step�_�lݾ       ��2	�r�̩�A�*

epsilon�_��FX.       ��W�	���̩�A�* 

Average reward per step�_���w`       ��2	���̩�A�*

epsilon�_��<@.       ��W�	YL�̩�A�* 

Average reward per step�_�,_�P       ��2	�L�̩�A�*

epsilon�_��Ere.       ��W�	en�̩�A�* 

Average reward per step�_�K�`       ��2	?o�̩�A�*

epsilon�_�=��.       ��W�	_]�̩�A�* 

Average reward per step�_���       ��2	�]�̩�A�*

epsilon�_�Ĉ�Y.       ��W�	�H�̩�A�* 

Average reward per step�_�0�       ��2	�I�̩�A�*

epsilon�_��d<�.       ��W�	�9�̩�A�* 

Average reward per step�_�=-�A       ��2	`:�̩�A�*

epsilon�_��8v�.       ��W�	?U�̩�A�* 

Average reward per step�_�f8��       ��2	�U�̩�A�*

epsilon�_�J�!.       ��W�	Z��̩�A�* 

Average reward per step�_���^f       ��2	훝̩�A�*

epsilon�_��/.       ��W�	���̩�A�* 

Average reward per step�_�����       ��2	`��̩�A�*

epsilon�_��en1.       ��W�	/��̩�A�* 

Average reward per step�_��x��       ��2	ۇ�̩�A�*

epsilon�_��h.       ��W�	ҧ�̩�A�* 

Average reward per step�_�W��.       ��2	���̩�A�*

epsilon�_��Ҹ1.       ��W�	s��̩�A�* 

Average reward per step�_��&�       ��2	,��̩�A�*

epsilon�_��-0       ���_	>Х̩�A*#
!
Average reward per episode��{��ZIL.       ��W�	�Х̩�A*!

total reward per episode  ��6�(l.       ��W�	��̩�A�* 

Average reward per step��{�!a��       ��2	��̩�A�*

epsilon��{�#���.       ��W�	�̩�A�* 

Average reward per step��{����       ��2	d�̩�A�*

epsilon��{�Y�.       ��W�	��̩�A�* 

Average reward per step��{�gW�       ��2	���̩�A�*

epsilon��{�yW-�.       ��W�	�"�̩�A�* 

Average reward per step��{�Nͬ       ��2	�#�̩�A�*

epsilon��{��2�".       ��W�	֩�̩�A�* 

Average reward per step��{���n�       ��2	ު�̩�A�*

epsilon��{�歧�.       ��W�	���̩�A�* 

Average reward per step��{�y>�"       ��2	��̩�A�*

epsilon��{��H�.       ��W�	?�̩�A�* 

Average reward per step��{��S�H       ��2	&�̩�A�*

epsilon��{��_�y.       ��W�	Q�̩�A�* 

Average reward per step��{��e��       ��2	+�̩�A�*

epsilon��{�ߢ�.       ��W�	�#�̩�A�* 

Average reward per step��{���"       ��2	y$�̩�A�*

epsilon��{��/rg.       ��W�	��̩�A�* 

Average reward per step��{�ʇ��       ��2	6�̩�A�*

epsilon��{�w9)�.       ��W�	��̩�A�* 

Average reward per step��{�g��       ��2	e�̩�A�*

epsilon��{�!��!.       ��W�	��̩�A�* 

Average reward per step��{��L��       ��2	y�̩�A�*

epsilon��{�t�2.       ��W�	�.�̩�A�* 

Average reward per step��{�]�M       ��2	�/�̩�A�*

epsilon��{��{�.       ��W�	|+�̩�A�* 

Average reward per step��{�A*�       ��2	0,�̩�A�*

epsilon��{��ݥ.       ��W�	�(�̩�A�* 

Average reward per step��{�T<�w       ��2	N)�̩�A�*

epsilon��{�!h�.       ��W�	�+�̩�A�* 

Average reward per step��{�P&<       ��2	�,�̩�A�*

epsilon��{����R.       ��W�	�E�̩�A�* 

Average reward per step��{��Ȼ�       ��2	EF�̩�A�*

epsilon��{�(�0.       ��W�	)=�̩�A�* 

Average reward per step��{�{�W^       ��2	�=�̩�A�*

epsilon��{���[5.       ��W�	�8�̩�A�* 

Average reward per step��{���       ��2	K9�̩�A�*

epsilon��{��`<�.       ��W�	�3�̩�A�* 

Average reward per step��{��*�       ��2	z4�̩�A�*

epsilon��{��[��.       ��W�	�9�̩�A�* 

Average reward per step��{�F	C       ��2	;:�̩�A�*

epsilon��{�)��m.       ��W�	&9�̩�A�* 

Average reward per step��{�U�?       ��2	�9�̩�A�*

epsilon��{���0       ���_	sg�̩�A*#
!
Average reward per episodeF���O��.       ��W�	�g�̩�A*!

total reward per episode  #�R0�Q.       ��W�	�_�̩�A�* 

Average reward per stepF��Q���       ��2	k`�̩�A�*

epsilonF��M�]�.       ��W�	f�̩�A�* 

Average reward per stepF��d�'o       ��2	�f�̩�A�*

epsilonF��҆�K.       ��W�	e�̩�A�* 

Average reward per stepF��e3T�       ��2	�e�̩�A�*

epsilonF��3ah�.       ��W�	��̩�A�* 

Average reward per stepF��ᢱ�       ��2	:�̩�A�*

epsilonF���r�.       ��W�	�t�̩�A�* 

Average reward per stepF����X�       ��2	�u�̩�A�*

epsilonF���1��.       ��W�	x~�̩�A�* 

Average reward per stepF���^0       ��2	J�̩�A�*

epsilonF����2.       ��W�	���̩�A�* 

Average reward per stepF��\x�6       ��2	���̩�A�*

epsilonF���IM.       ��W�	���̩�A�* 

Average reward per stepF�����       ��2	:��̩�A�*

epsilonF���y�C.       ��W�	�H�̩�A�* 

Average reward per stepF�����       ��2	�I�̩�A�*

epsilonF���.       ��W�	���̩�A�* 

Average reward per stepF��:�       ��2	l��̩�A�*

epsilonF���.       ��W�	;��̩�A�* 

Average reward per stepF��sD��       ��2	*��̩�A�*

epsilonF��Sp:�.       ��W�	I��̩�A�* 

Average reward per stepF��ib��       ��2	���̩�A�*

epsilonF���u^.       ��W�	��̩�A�* 

Average reward per stepF���{�       ��2	���̩�A�*

epsilonF��ɐZ0       ���_	�"�̩�A	*#
!
Average reward per episode��N�����.       ��W�	l#�̩�A	*!

total reward per episode  (öqG�.       ��W�	�)�̩�A�* 

Average reward per step��N�Aj�       ��2	�*�̩�A�*

epsilon��N�8uS�.       ��W�	
J�̩�A�* 

Average reward per step��N�8��       ��2	�J�̩�A�*

epsilon��N���V�.       ��W�	ZJ�̩�A�* 

Average reward per step��N��d��       ��2	�J�̩�A�*

epsilon��N��,.       ��W�	�C�̩�A�* 

Average reward per step��N�#^*�       ��2	�D�̩�A�*

epsilon��N�/��.       ��W�	�R�̩�A�* 

Average reward per step��N��|       ��2	�S�̩�A�*

epsilon��N���&�.       ��W�	�N ̩�A�* 

Average reward per step��N��!�       ��2	�O ̩�A�*

epsilon��N��ٌ�.       ��W�	�N̩�A�* 

Average reward per step��N�@�       ��2	O̩�A�*

epsilon��N�R{W�.       ��W�	�n̩�A�* 

Average reward per step��N��J��       ��2	�o̩�A�*

epsilon��N���e�.       ��W�	��̩�A�* 

Average reward per step��N�v��       ��2	\�̩�A�*

epsilon��N�~E��.       ��W�	Ί̩�A�* 

Average reward per step��N��E`       ��2	z�̩�A�*

epsilon��N�dc�.       ��W�	��
̩�A�* 

Average reward per step��N���e�       ��2	E�
̩�A�*

epsilon��N�N��%.       ��W�	��̩�A�* 

Average reward per step��N��-��       ��2	��̩�A�*

epsilon��N��w�.       ��W�	3�̩�A�* 

Average reward per step��N���r�       ��2		�̩�A�*

epsilon��N��C��.       ��W�	�̩�A�* 

Average reward per step��N�<�ϝ       ��2	�̩�A�*

epsilon��N�1�w�.       ��W�	ծ̩�A�* 

Average reward per step��N��A��       ��2	��̩�A�*

epsilon��N���R�.       ��W�	1^̩�A�* 

Average reward per step��N��P�       ��2	_̩�A�*

epsilon��N��͉.       ��W�	��̩�A�* 

Average reward per step��N�D<       ��2	ȱ̩�A�*

epsilon��N�~��f.       ��W�	%�̩�A�* 

Average reward per step��N�;�-+       ��2	��̩�A�*

epsilon��N��Sx.       ��W�	��̩�A�* 

Average reward per step��N� �R       ��2	��̩�A�*

epsilon��N���m0       ���_	�̩�A
*#
!
Average reward per episode���~�7.       ��W�	>̩�A
*!

total reward per episode  &�Q�?�.       ��W�	�8̩�A�* 

Average reward per step��%Z�       ��2	�9̩�A�*

epsilon��JGu*.       ��W�	\:!̩�A�* 

Average reward per step�����
       ��2	 ;!̩�A�*

epsilon��49�.       ��W�	�"̩�A�* 

Average reward per step����-<       ��2	Ւ"̩�A�*

epsilon��v�2�.       ��W�	M�#̩�A�* 

Average reward per step��oL       ��2	�#̩�A�*

epsilon��^Lj�.       ��W�	�&̩�A�* 

Average reward per step����2       ��2	�&̩�A�*

epsilon���i.       ��W�	(̩�A�* 

Average reward per step�����       ��2	�(̩�A�*

epsilon�����.       ��W�	�%*̩�A�* 

Average reward per step��P#��       ��2	'*̩�A�*

epsilon��3��A.       ��W�	�C,̩�A�* 

Average reward per step���WT�       ��2	�D,̩�A�*

epsilon��l���.       ��W�	TV.̩�A�* 

Average reward per step����GN       ��2	�V.̩�A�*

epsilon���+u?.       ��W�	͓0̩�A�* 

Average reward per step��{O�       ��2	W�0̩�A�*

epsilon��L��.       ��W�	�2̩�A�* 

Average reward per step���|��       ��2	��2̩�A�*

epsilon���Ai.       ��W�	�4̩�A�* 

Average reward per step��	b�       ��2	��4̩�A�*

epsilon��!�L�.       ��W�	t�6̩�A�* 

Average reward per step�����\       ��2	�6̩�A�*

epsilon���"�+.       ��W�	�'8̩�A�* 

Average reward per step���ʈ�       ��2	�(8̩�A�*

epsilon���<<�.       ��W�	�N:̩�A�* 

Average reward per step��*���       ��2	O:̩�A�*

epsilon���H��.       ��W�	�E<̩�A�* 

Average reward per step��H3!       ��2	�F<̩�A�*

epsilon��CT�.       ��W�	ۋ=̩�A�* 

Average reward per step��<��n       ��2	��=̩�A�*

epsilon��j��.       ��W�	��?̩�A�* 

Average reward per step����̫       ��2	j�?̩�A�*

epsilon���p4S.       ��W�	��A̩�A�* 

Average reward per step���`|       ��2	׊A̩�A�*

epsilon��M�ש.       ��W�	�yC̩�A�* 

Average reward per step����5       ��2	}zC̩�A�*

epsilon���ߥ.       ��W�	��E̩�A�* 

Average reward per step���͘�       ��2	w�E̩�A�*

epsilon���*�.       ��W�	��G̩�A�* 

Average reward per step��*֎b       ��2	w�G̩�A�*

epsilon��h��.       ��W�	�I̩�A�* 

Average reward per step�����       ��2	��I̩�A�*

epsilon��vbk.       ��W�	��K̩�A�* 

Average reward per step��f��       ��2	B�K̩�A�*

epsilon��j���.       ��W�	��M̩�A�* 

Average reward per step���`�       ��2	��M̩�A�*

epsilon����d.       ��W�	�O̩�A�* 

Average reward per step��mV�       ��2	��O̩�A�*

epsilon����Ņ.       ��W�	�Q̩�A�* 

Average reward per step��	[M�       ��2	��Q̩�A�*

epsilon��2a �.       ��W�	.S̩�A�* 

Average reward per step��Ds~�       ��2	Q/S̩�A�*

epsilon���%�.       ��W�	;U̩�A�* 

Average reward per step�����       ��2	)<U̩�A�*

epsilon��6M �.       ��W�	kW̩�A�* 

Average reward per step�����n       ��2	3lW̩�A�*

epsilon��&Gw�.       ��W�	��Y̩�A�* 

Average reward per step���Y�       ��2	�Y̩�A�*

epsilon��܁�.       ��W�	�[̩�A�* 

Average reward per step���Pa       ��2	�[̩�A�*

epsilon���R�.       ��W�	�M\̩�A�* 

Average reward per step���}<}       ��2	�N\̩�A�*

epsilon��i��.       ��W�	�9^̩�A�* 

Average reward per step�����A       ��2	�;^̩�A�*

epsilon��x�w0       ���_	 V^̩�A*#
!
Average reward per episodeo�O
K�.       ��W�	�V^̩�A*!

total reward per episode  ���8.       ��W�	ab̩�A�* 

Average reward per stepo�H���       ��2	�ab̩�A�*

epsilono���و.       ��W�	(bd̩�A�* 

Average reward per stepo�S>��       ��2	cd̩�A�*

epsilono��!��.       ��W�	"qf̩�A�* 

Average reward per stepo�7p��       ��2	�qf̩�A�*

epsilono�%��.       ��W�	�rh̩�A�* 

Average reward per stepo��	W       ��2	�sh̩�A�*

epsilono�Ui.       ��W�	Tqj̩�A�* 

Average reward per stepo���x�       ��2	rj̩�A�*

epsilono���EP.       ��W�	C�l̩�A�* 

Average reward per stepo�0��       ��2	�l̩�A�*

epsilono��ZN.       ��W�	�n̩�A�* 

Average reward per stepo����       ��2	��n̩�A�*

epsilono�Yu$7.       ��W�	'�p̩�A�* 

Average reward per stepo�7Y        ��2	�p̩�A�*

epsilono��Y�/.       ��W�	V�r̩�A�* 

Average reward per stepo���G       ��2	��r̩�A�*

epsilono�\r>.       ��W�	R�t̩�A�* 

Average reward per stepo���g�       ��2	��t̩�A�*

epsilono�u>�#.       ��W�	�v̩�A�* 

Average reward per stepo�&z��       ��2	��v̩�A�*

epsilono��ˑ�.       ��W�	:�x̩�A�* 

Average reward per stepo�r��       ��2	%�x̩�A�*

epsilono�'|��.       ��W�	^{̩�A�* 

Average reward per stepo�/iE`       ��2	+{̩�A�*

epsilono�\^��.       ��W�	h}̩�A�* 

Average reward per stepo�Z*�       ��2	�}̩�A�*

epsilono�#/}.       ��W�	�̩�A�* 

Average reward per stepo�m]<�       ��2	�̩�A�*

epsilono���.       ��W�	�J�̩�A�* 

Average reward per stepo��)�J       ��2	�K�̩�A�*

epsilono��ֿ\.       ��W�	�E�̩�A�* 

Average reward per stepo�����       ��2	�F�̩�A�*

epsilono�D͵=.       ��W�	�?�̩�A�* 

Average reward per stepo� �=�       ��2	�@�̩�A�*

epsilono�����0       ���_	k�̩�A*#
!
Average reward per episodeUU��>��.       ��W�	�k�̩�A*!

total reward per episode  �_]�.       ��W�	D߈̩�A�* 

Average reward per stepUU�7 de       ��2	�߈̩�A�*

epsilonUU�>��x.       ��W�	2�̩�A�* 

Average reward per stepUU�Z�_       ��2	��̩�A�*

epsilonUU��hv�.       ��W�	�&�̩�A�* 

Average reward per stepUU���       ��2	�'�̩�A�*

epsilonUU�/*.       ��W�	�-�̩�A�* 

Average reward per stepUU��       ��2	�.�̩�A�*

epsilonUU�,V�.       ��W�	>@�̩�A�* 

Average reward per stepUU����G       ��2	�@�̩�A�*

epsilonUU��h��.       ��W�	�m�̩�A�* 

Average reward per stepUU��`"       ��2	�n�̩�A�*

epsilonUU�3�m�.       ��W�	ۤ�̩�A�* 

Average reward per stepUU�i�|�       ��2	���̩�A�*

epsilonUU�y�?
.       ��W�	k��̩�A�* 

Average reward per stepUU��       ��2	��̩�A�*

epsilonUU�E9~�.       ��W�	P�̩�A�* 

Average reward per stepUU���~�       ��2	�P�̩�A�*

epsilonUU�i�c.       ��W�	쇚̩�A�* 

Average reward per stepUU����       ��2	���̩�A�*

epsilonUU��r2.       ��W�	N&�̩�A�* 

Average reward per stepUU�9�8}       ��2	('�̩�A�*

epsilonUU���%.       ��W�	�Ξ̩�A�* 

Average reward per stepUU����       ��2	hϞ̩�A�*

epsilonUU�.       ��W�	�Р̩�A�* 

Average reward per stepUU��l�       ��2	Ӡ̩�A�*

epsilonUU���"3.       ��W�	�o�̩�A�* 

Average reward per stepUU�z8�       ��2	�p�̩�A�*

epsilonUU�����.       ��W�	wh�̩�A�* 

Average reward per stepUU����       ��2	i�̩�A�*

epsilonUU�j��`.       ��W�	^��̩�A�* 

Average reward per stepUU��p�2       ��2	���̩�A�*

epsilonUU�g�n.       ��W�	Ӆ�̩�A�* 

Average reward per stepUU��=       ��2	���̩�A�*

epsilonUU���W0       ���_	��̩�A*#
!
Average reward per episodeZZ�N�<.       ��W�	���̩�A*!

total reward per episode  $���a.       ��W�	O��̩�A�* 

Average reward per stepZZ��7w       ��2	2��̩�A�*

epsilonZZ�vM��.       ��W�	׮̩�A�* 

Average reward per stepZZ�ݮ:&       ��2	�׮̩�A�*

epsilonZZ�J�R.       ��W�	cհ̩�A�* 

Average reward per stepZZ��0�s       ��2	,ְ̩�A�*

epsilonZZ�%�*
.       ��W�	�̲̩�A�* 

Average reward per stepZZ���S       ��2	�Ͳ̩�A�*

epsilonZZ����.       ��W�	��̩�A�* 

Average reward per stepZZ���       ��2	u�̩�A�*

epsilonZZ����.       ��W�	��̩�A�* 

Average reward per stepZZ���e�       ��2	\�̩�A�*

epsilonZZ��e�`.       ��W�	WѸ̩�A�* 

Average reward per stepZZ��~��       ��2	Ҹ̩�A�*

epsilonZZ��Oc(.       ��W�	p�̩�A�* 

Average reward per stepZZ��d��       ��2	�̩�A�*

epsilonZZ���j.       ��W�	���̩�A�* 

Average reward per stepZZ��n8�       ��2	��̩�A�*

epsilonZZ��Z�.       ��W�	���̩�A�* 

Average reward per stepZZ�����       ��2	���̩�A�*

epsilonZZ�S�.       ��W�	��̩�A�* 

Average reward per stepZZ�sղ       ��2	���̩�A�*

epsilonZZ�:�/.       ��W�	��̩�A�* 

Average reward per stepZZ���H�       ��2	0�̩�A�*

epsilonZZ��Q�h.       ��W�	��̩�A�* 

Average reward per stepZZ��8�       ��2	g�̩�A�*

epsilonZZ���2x.       ��W�	 �̩�A�* 

Average reward per stepZZ���       ��2	�̩�A�*

epsilonZZ���fh.       ��W�	�4�̩�A�* 

Average reward per stepZZ�_͟�       ��2	�5�̩�A�*

epsilonZZ��'<�.       ��W�	<3�̩�A�* 

Average reward per stepZZ�X]��       ��2	4�̩�A�*

epsilonZZ���hd.       ��W�	���̩�A�* 

Average reward per stepZZ�~-,j       ��2	���̩�A�*

epsilonZZ���Qq.       ��W�	}��̩�A�* 

Average reward per stepZZ�G�5       ��2	!��̩�A�*

epsilonZZ�Z.��.       ��W�	���̩�A�* 

Average reward per stepZZ��w�q       ��2	���̩�A�*

epsilonZZ��!�.       ��W�	���̩�A�* 

Average reward per stepZZ�s¨�       ��2	ʥ�̩�A�*

epsilonZZ��M�.       ��W�	��̩�A�* 

Average reward per stepZZ���(�       ��2	~��̩�A�*

epsilonZZ�;sb�0       ���_	���̩�A*#
!
Average reward per episodez�����",.       ��W�	h��̩�A*!

total reward per episode  �&IK[.       ��W�	���̩�A�* 

Average reward per stepz���#�x�       ��2	���̩�A�*

epsilonz����/d�.       ��W�	���̩�A�* 

Average reward per stepz������       ��2	���̩�A�*

epsilonz���BM^.       ��W�	���̩�A�* 

Average reward per stepz���=?uj       ��2	���̩�A�*

epsilonz���� g$.       ��W�	���̩�A�* 

Average reward per stepz����'�a       ��2	���̩�A�*

epsilonz���i�.       ��W�	�
�̩�A�* 

Average reward per stepz������       ��2	��̩�A�*

epsilonz���$m�2.       ��W�	��̩�A�* 

Average reward per stepz���~��r       ��2	���̩�A�*

epsilonz���Co�'.       ��W�	 ��̩�A�* 

Average reward per stepz���+$Q        ��2	��̩�A�*

epsilonz����.       ��W�	'��̩�A�* 

Average reward per stepz����m5�       ��2	��̩�A�*

epsilonz����2x.       ��W�	�T�̩�A�* 

Average reward per stepz���yL��       ��2	\U�̩�A�*

epsilonz����S�.       ��W�	d��̩�A�* 

Average reward per stepz����ȋ�       ��2	:��̩�A�*

epsilonz���%Q�.       ��W�	�M�̩�A�* 

Average reward per stepz���A��       ��2	DN�̩�A�*

epsilonz���yJ=�.       ��W�	���̩�A�* 

Average reward per stepz���E4�.       ��2	4��̩�A�*

epsilonz����¥�.       ��W�	6��̩�A�* 

Average reward per stepz����uQ       ��2	���̩�A�*

epsilonz���в;�.       ��W�	L��̩�A�* 

Average reward per stepz����V��       ��2	���̩�A�*

epsilonz���:��.       ��W�	-\�̩�A�* 

Average reward per stepz������l       ��2	�\�̩�A�*

epsilonz������X.       ��W�	 ��̩�A�* 

Average reward per stepz�����<�       ��2	���̩�A�*

epsilonz������.       ��W�	���̩�A�* 

Average reward per stepz���u�(       ��2	A��̩�A�*

epsilonz����S.       ��W�	M��̩�A�* 

Average reward per stepz����B��       ��2	���̩�A�*

epsilonz���x�	.       ��W�	���̩�A�* 

Average reward per stepz���DO�A       ��2	Y��̩�A�*

epsilonz���9�Y�.       ��W�	���̩�A�* 

Average reward per stepz����ցM       ��2	V��̩�A�*

epsilonz����	%�.       ��W�	A ̩�A�* 

Average reward per stepz���4���       ��2	�A ̩�A�*

epsilonz����zH�.       ��W�	n�̩�A�* 

Average reward per stepz������+       ��2	7�̩�A�*

epsilonz����~F�.       ��W�	Έ̩�A�* 

Average reward per stepz���O|f�       ��2	��̩�A�*

epsilonz�������.       ��W�	�r̩�A�* 

Average reward per stepz���u��t       ��2	�s̩�A�*

epsilonz���Q�l.       ��W�	7q̩�A�* 

Average reward per stepz���(>s�       ��2	r̩�A�*

epsilonz����=h.       ��W�	�w
̩�A�* 

Average reward per stepz���@o��       ��2	Fx
̩�A�*

epsilonz������.       ��W�	 o̩�A�* 

Average reward per stepz������Y       ��2	�o̩�A�*

epsilonz���}-��.       ��W�	��̩�A�* 

Average reward per stepz���7�f�       ��2	�̩�A�*

epsilonz���#}l .       ��W�	�̩�A�* 

Average reward per stepz���!1�       ��2	ȴ̩�A�*

epsilonz���rV��.       ��W�	�̩�A�* 

Average reward per stepz����s�       ��2	۾̩�A�*

epsilonz���AX�.       ��W�	��̩�A�* 

Average reward per stepz�����u       ��2	��̩�A�*

epsilonz��� -�.       ��W�		R̩�A�* 

Average reward per stepz�����+�       ��2	�R̩�A�*

epsilonz�����".       ��W�	r�̩�A�* 

Average reward per stepz���6/��       ��2	r�̩�A�*

epsilonz�����5.       ��W�	�̩�A�* 

Average reward per stepz�����       ��2	��̩�A�*

epsilonz����m�.       ��W�	RD̩�A�* 

Average reward per stepz�����i�       ��2	�F̩�A�*

epsilonz���垆.       ��W�	�̩�A�* 

Average reward per stepz����V�       ��2	��̩�A�*

epsilonz���V���.       ��W�	k�̩�A�* 

Average reward per stepz�����)       ��2	R�̩�A�*

epsilonz���Sh�.       ��W�	%!̩�A�* 

Average reward per stepz����$*       ��2	 !̩�A�*

epsilonz�������.       ��W�	�i"̩�A�* 

Average reward per stepz����g$�       ��2	�j"̩�A�*

epsilonz����+�.       ��W�	��#̩�A�* 

Average reward per stepz����z�       ��2	L�#̩�A�*

epsilonz����(2.       ��W�	�%̩�A�* 

Average reward per stepz���>py0       ��2	��%̩�A�*

epsilonz���,2�+.       ��W�	'l(̩�A�* 

Average reward per stepz���BP{~       ��2	&m(̩�A�*

epsilonz����a�.       ��W�	�)̩�A�* 

Average reward per stepz����ra       ��2	��)̩�A�*

epsilonz���ʾ�6.       ��W�	 ,̩�A�* 

Average reward per stepz�����k�       ��2	�,̩�A�*

epsilonz���⽣2.       ��W�	|,.̩�A�* 

Average reward per stepz���G��       ��2	k-.̩�A�*

epsilonz����yj�.       ��W�	�10̩�A�* 

Average reward per stepz���:u�       ��2		30̩�A�*

epsilonz����@,�.       ��W�	v32̩�A�* 

Average reward per stepz���g��       ��2	42̩�A�*

epsilonz���T��.       ��W�	��3̩�A�* 

Average reward per stepz������
       ��2	��3̩�A�*

epsilonz����I�.       ��W�	t#5̩�A�* 

Average reward per stepz����xQ�       ��2	t$5̩�A�*

epsilonz�����s�.       ��W�	nL7̩�A�* 

Average reward per stepz���ܫ�       ��2	8M7̩�A�*

epsilonz����].       ��W�	�G9̩�A�* 

Average reward per stepz������       ��2	�H9̩�A�*

epsilonz������.       ��W�	�J;̩�A�* 

Average reward per stepz�����_q       ��2	�L;̩�A�*

epsilonz������ .       ��W�	�?=̩�A�* 

Average reward per stepz���6v�       ��2	J@=̩�A�*

epsilonz���hVY�.       ��W�	�(?̩�A�* 

Average reward per stepz���}�H       ��2	�)?̩�A�*

epsilonz�������.       ��W�	(A̩�A�* 

Average reward per stepz���#j�       ��2	�(A̩�A�*

epsilonz�����0       ���_	�AA̩�A*#
!
Average reward per episodek�0���9.       ��W�	1BA̩�A*!

total reward per episode  �T��!.       ��W�	�VE̩�A�* 

Average reward per stepk�0����       ��2	�WE̩�A�*

epsilonk�0��'�.       ��W�	�FG̩�A�* 

Average reward per stepk�0���Cb       ��2	gGG̩�A�*

epsilonk�0��g��.       ��W�	�
J̩�A�* 

Average reward per stepk�0�w�y{       ��2	tJ̩�A�*

epsilonk�0��ן.       ��W�	XtM̩�A�* 

Average reward per stepk�0����       ��2	uM̩�A�*

epsilonk�0���.       ��W�	_O̩�A�* 

Average reward per stepk�0�����       ��2	$�O̩�A�*

epsilonk�0���?.       ��W�	�mQ̩�A�* 

Average reward per stepk�0��K��       ��2	�nQ̩�A�*

epsilonk�0���.       ��W�	��S̩�A�* 

Average reward per stepk�0�#z>       ��2	��S̩�A�*

epsilonk�0����.       ��W�	�^W̩�A�* 

Average reward per stepk�0� 4��       ��2	�_W̩�A�*

epsilonk�0���.       ��W�	2�Y̩�A�* 

Average reward per stepk�0��@��       ��2	6�Y̩�A�*

epsilonk�0�1f��.       ��W�	��[̩�A�* 

Average reward per stepk�0�˾7�       ��2	k�[̩�A�*

epsilonk�0��sv�.       ��W�	�]̩�A�* 

Average reward per stepk�0�� ��       ��2	ѭ]̩�A�*

epsilonk�0��c>C.       ��W�	��_̩�A�* 

Average reward per stepk�0��f��       ��2	ˢ_̩�A�*

epsilonk�0�e��.       ��W�	��a̩�A�* 

Average reward per stepk�0��J]       ��2	:�a̩�A�*

epsilonk�0���.       ��W�	U�c̩�A�* 

Average reward per stepk�0���p       ��2	7�c̩�A�*

epsilonk�0�@�.       ��W�	a�e̩�A�* 

Average reward per stepk�0��mB       ��2	3�e̩�A�*

epsilonk�0�-��.       ��W�	S�g̩�A�* 

Average reward per stepk�0�p!4�       ��2	F�g̩�A�*

epsilonk�0�j_�'.       ��W�	�j̩�A�* 

Average reward per stepk�0�q�L�       ��2	lj̩�A�*

epsilonk�0����.       ��W�	Cl̩�A�* 

Average reward per stepk�0���       ��2	�Cl̩�A�*

epsilonk�0���k�.       ��W�	�Mn̩�A�* 

Average reward per stepk�0���       ��2	bNn̩�A�*

epsilonk�0�OY��.       ��W�	�r̩�A�* 

Average reward per stepk�0��K[       ��2	�r̩�A�*

epsilonk�0�3�.�.       ��W�	H4t̩�A�* 

Average reward per stepk�0��v       ��2	+5t̩�A�*

epsilonk�0��Ș�.       ��W�	��v̩�A�* 

Average reward per stepk�0��_8�       ��2	��v̩�A�*

epsilonk�0��([.       ��W�	�<x̩�A�* 

Average reward per stepk�0���`       ��2	�=x̩�A�*

epsilonk�0���#.       ��W�	�Vz̩�A�* 

Average reward per stepk�0��{7       ��2	GWz̩�A�*

epsilonk�0�sׁ�.       ��W�	�V|̩�A�* 

Average reward per stepk�0����       ��2	�W|̩�A�*

epsilonk�0����.       ��W�	�w~̩�A�* 

Average reward per stepk�0�~�T\       ��2	ux~̩�A�*

epsilonk�0��l�.       ��W�	���̩�A�* 

Average reward per stepk�0��gD6       ��2	z��̩�A�*

epsilonk�0�a�?�0       ���_	��̩�A*#
!
Average reward per episode�K������.       ��W�	ꮀ̩�A*!

total reward per episode  �ƐT�.       ��W�	�%�̩�A�* 

Average reward per step�K��U�       ��2	�&�̩�A�*

epsilon�K��Ԃ
�.       ��W�	�}�̩�A�* 

Average reward per step�K���ƙ�       ��2	�~�̩�A�*

epsilon�K�����.       ��W�	ƨ�̩�A�* 

Average reward per step�K����       ��2	i��̩�A�*

epsilon�K��u�1T.       ��W�	l�̩�A�* 

Average reward per step�K��9��       ��2	K�̩�A�*

epsilon�K���ihP.       ��W�	�b�̩�A�* 

Average reward per step�K����+{       ��2	�c�̩�A�*

epsilon�K��C��{.       ��W�	${�̩�A�* 

Average reward per step�K���k�       ��2	�{�̩�A�*

epsilon�K���Z3�.       ��W�	S{�̩�A�* 

Average reward per step�K���Y2�       ��2	�{�̩�A�*

epsilon�K���X�.       ��W�	���̩�A�* 

Average reward per step�K��x���       ��2	u��̩�A�*

epsilon�K���t��.       ��W�	ۉ�̩�A�* 

Average reward per step�K����K       ��2	���̩�A�*

epsilon�K��"O��.       ��W�	���̩�A�* 

Average reward per step�K��ʤ6�       ��2	P��̩�A�*

epsilon�K���9�.       ��W�	p�̩�A�* 

Average reward per step�K��q��k       ��2	!�̩�A�*

epsilon�K��W��G.       ��W�	�*�̩�A�* 

Average reward per step�K����W       ��2	�+�̩�A�*

epsilon�K��{J�t.       ��W�	ҏ�̩�A�* 

Average reward per step�K���T��       ��2	���̩�A�*

epsilon�K��{<n.       ��W�	ݛ̩�A�* 

Average reward per step�K��l�J       ��2	�ݛ̩�A�*

epsilon�K���ZX.       ��W�	��̩�A�* 

Average reward per step�K�����       ��2	`�̩�A�*

epsilon�K���r�J.       ��W�	"��̩�A�* 

Average reward per step�K��W�>S       ��2	���̩�A�*

epsilon�K���P.�.       ��W�	�̩�A�* 

Average reward per step�K��D�`       ��2	��̩�A�*

epsilon�K���g^#.       ��W�	�"�̩�A�* 

Average reward per step�K��r�R�       ��2	t#�̩�A�*

epsilon�K�����.       ��W�	"�̩�A�* 

Average reward per step�K���(�       ��2	��̩�A�*

epsilon�K��x�|.       ��W�	� �̩�A�* 

Average reward per step�K��N��       ��2	}!�̩�A�*

epsilon�K����.       ��W�	꒪̩�A�* 

Average reward per step�K��kJ�3       ��2	���̩�A�*

epsilon�K���� .       ��W�	,�̩�A�* 

Average reward per step�K�����       ��2	�,�̩�A�*

epsilon�K���Y��.       ��W�	(�̩�A�* 

Average reward per step�K���Lp�       ��2	�(�̩�A�*

epsilon�K�����v.       ��W�	��̩�A�* 

Average reward per step�K���,Ɓ       ��2	E��̩�A�*

epsilon�K������.       ��W�	Oʱ̩�A�* 

Average reward per step�K���5��       ��2	S˱̩�A�*

epsilon�K���^�@.       ��W�	v�̩�A�* 

Average reward per step�K��}�#�       ��2	�v�̩�A�*

epsilon�K��h��.       ��W�	X��̩�A�* 

Average reward per step�K���"<�       ��2	?��̩�A�*

epsilon�K��F ѹ.       ��W�	5��̩�A�* 

Average reward per step�K���VP       ��2	��̩�A�*

epsilon�K����zS.       ��W�	�չ̩�A�* 

Average reward per step�K���]�V       ��2	�ֹ̩�A�*

epsilon�K��HT�=.       ��W�	�7�̩�A�* 

Average reward per step�K��7VA       ��2	�8�̩�A�*

epsilon�K�����.       ��W�	�ܽ̩�A�* 

Average reward per step�K��w��       ��2	�ݽ̩�A�*

epsilon�K���n8.       ��W�	X�̩�A�* 

Average reward per step�K��]���       ��2	%�̩�A�*

epsilon�K����L�.       ��W�	�}�̩�A�* 

Average reward per step�K��\1O       ��2	R~�̩�A�*

epsilon�K���p]�.       ��W�	���̩�A�* 

Average reward per step�K���J<!       ��2	|��̩�A�*

epsilon�K����q.       ��W�	a��̩�A�* 

Average reward per step�K���!       ��2	P �̩�A�*

epsilon�K���2�.       ��W�	�
�̩�A�* 

Average reward per step�K�����       ��2	 �̩�A�*

epsilon�K��'F.       ��W�	� �̩�A�* 

Average reward per step�K���.z:       ��2	m�̩�A�*

epsilon�K��t�.       ��W�	$~�̩�A�* 

Average reward per step�K��tʲ�       ��2	�~�̩�A�*

epsilon�K��B~n.       ��W�	��̩�A�* 

Average reward per step�K���k��       ��2	͏�̩�A�*

epsilon�K��j�<�.       ��W�	ܼ�̩�A�* 

Average reward per step�K��?��q       ��2	���̩�A�*

epsilon�K��c�s�.       ��W�	�8�̩�A�* 

Average reward per step�K��.DV       ��2	K9�̩�A�*

epsilon�K��H��:.       ��W�	�/�̩�A�* 

Average reward per step�K��r-�D       ��2	/0�̩�A�*

epsilon�K���7�.       ��W�	>��̩�A�* 

Average reward per step�K�� e
L       ��2	5��̩�A�*

epsilon�K����'}.       ��W�	q�̩�A�* 

Average reward per step�K���f�       ��2	�q�̩�A�*

epsilon�K��ben&.       ��W�	�t�̩�A�* 

Average reward per step�K�� ���       ��2	Ku�̩�A�*

epsilon�K���CD.       ��W�	�s�̩�A�* 

Average reward per step�K��\չ       ��2	�t�̩�A�*

epsilon�K����.       ��W�	Ƌ�̩�A�* 

Average reward per step�K��'~��       ��2	���̩�A�*

epsilon�K���|�[.       ��W�	ƣ�̩�A�* 

Average reward per step�K���R%�       ��2	Ӥ�̩�A�*

epsilon�K����*�.       ��W�	��̩�A�* 

Average reward per step�K���&��       ��2	���̩�A�*

epsilon�K��Ev	�.       ��W�	���̩�A�* 

Average reward per step�K��lM
F       ��2	���̩�A�*

epsilon�K��xm&.       ��W�	���̩�A�* 

Average reward per step�K��y��C       ��2	/��̩�A�*

epsilon�K�����.       ��W�	���̩�A�* 

Average reward per step�K����0       ��2	=��̩�A�*

epsilon�K���)�=.       ��W�	���̩�A�* 

Average reward per step�K����ۯ       ��2	W��̩�A�*

epsilon�K��Wn��.       ��W�	A��̩�A�* 

Average reward per step�K��m��       ��2	I��̩�A�*

epsilon�K��XQc�.       ��W�	�̩�A�* 

Average reward per step�K�����       ��2	��̩�A�*

epsilon�K���Ƒ.       ��W�	�*�̩�A�* 

Average reward per step�K���V�k       ��2	+�̩�A�*

epsilon�K��F?�^.       ��W�	-�̩�A�* 

Average reward per step�K���Q�       ��2	�-�̩�A�*

epsilon�K��:��	.       ��W�	�8�̩�A�* 

Average reward per step�K���`"       ��2	�9�̩�A�*

epsilon�K���rJ�.       ��W�	_F�̩�A�* 

Average reward per step�K���k       ��2	G�̩�A�*

epsilon�K��#7�.       ��W�	�E�̩�A�* 

Average reward per step�K����(       ��2	�F�̩�A�*

epsilon�K���y(�.       ��W�	�T�̩�A�* 

Average reward per step�K���B.Y       ��2	�U�̩�A�*

epsilon�K���_��.       ��W�	��̩�A�* 

Average reward per step�K���s�       ��2	���̩�A�*

epsilon�K���rr�.       ��W�	��̩�A�* 

Average reward per step�K��o��       ��2	z�̩�A�*

epsilon�K��u��.       ��W�	O�̩�A�* 

Average reward per step�K���[S�       ��2	�̩�A�*

epsilon�K����L�.       ��W�	�̩�A�* 

Average reward per step�K�����L       ��2	��̩�A�*

epsilon�K��Mn.       ��W�	�̩�A�* 

Average reward per step�K���->a       ��2	�̩�A�*

epsilon�K��̀$.       ��W�	j�	̩�A�* 

Average reward per step�K��6� �       ��2	�	̩�A�*

epsilon�K���.W!.       ��W�	}�̩�A�* 

Average reward per step�K��(h/[       ��2	>�̩�A�*

epsilon�K�� \�-.       ��W�	�̩�A�* 

Average reward per step�K��C�X�       ��2	�̩�A�*

epsilon�K��9c�.       ��W�	�Y̩�A�* 

Average reward per step�K����       ��2	�Z̩�A�*

epsilon�K��q��.       ��W�	L�̩�A�* 

Average reward per step�K��H�[       ��2	;�̩�A�*

epsilon�K��<j0       ���_	`̩�A*#
!
Average reward per episodeH�B��hj.       ��W�	!̩�A*!

total reward per episode  X�|&J.       ��W�	b̩�A�* 

Average reward per stepH�B����k       ��2	�b̩�A�*

epsilonH�B���a1.       ��W�	5	̩�A�* 

Average reward per stepH�B�e��X       ��2	 
̩�A�*

epsilonH�B�Hxh�.       ��W�	̩�A�* 

Average reward per stepH�B�QR׸       ��2	�̩�A�*

epsilonH�B�׍�.       ��W�	w�̩�A�* 

Average reward per stepH�B�
�       ��2	Q�̩�A�*

epsilonH�B���VB.       ��W�	��̩�A�* 

Average reward per stepH�B�Rv��       ��2	L�̩�A�*

epsilonH�B�� �.       ��W�	��̩�A�* 

Average reward per stepH�B�F*�       ��2	��̩�A�*

epsilonH�B�&\.       ��W�	�̩�A�* 

Average reward per stepH�B���~       ��2	�̩�A�*

epsilonH�B��`.       ��W�	�1!̩�A�* 

Average reward per stepH�B���(       ��2	v2!̩�A�*

epsilonH�B�*Nt�.       ��W�	L6#̩�A�* 

Average reward per stepH�B�o5�N       ��2	7#̩�A�*

epsilonH�B�����.       ��W�	�8%̩�A�* 

Average reward per stepH�B�W�"       ��2	�9%̩�A�*

epsilonH�B��Z2�.       ��W�	p'̩�A�* 

Average reward per stepH�B��R#       ��2	�p'̩�A�*

epsilonH�B�-���.       ��W�	})̩�A�* 

Average reward per stepH�B�(ۤt       ��2	�})̩�A�*

epsilonH�B�z��.       ��W�	w�+̩�A�* 

Average reward per stepH�B��`��       ��2	^�+̩�A�*

epsilonH�B����.       ��W�	CW-̩�A�* 

Average reward per stepH�B�Y���       ��2	*X-̩�A�*

epsilonH�B�k���.       ��W�	20̩�A�* 

Average reward per stepH�B�Ů�       ��2	�0̩�A�*

epsilonH�B� DG.       ��W�	f02̩�A�* 

Average reward per stepH�B�r$!#       ��2	Q12̩�A�*

epsilonH�B��2��.       ��W�	]�5̩�A�* 

Average reward per stepH�B�d^�       ��2	D�5̩�A�*

epsilonH�B�M�v.       ��W�	��7̩�A�* 

Average reward per stepH�B��?":       ��2	��7̩�A�*

epsilonH�B��Ҕ.       ��W�	Z9̩�A�* 

Average reward per stepH�B�<Ă=       ��2	؀9̩�A�*

epsilonH�B���=.       ��W�	��;̩�A�* 

Average reward per stepH�B���       ��2	��;̩�A�*

epsilonH�B���"�.       ��W�	(�=̩�A�* 

Average reward per stepH�B��"^       ��2	Ϻ=̩�A�*

epsilonH�B�Ӑ[.       ��W�	k-?̩�A�* 

Average reward per stepH�B����       ��2	E.?̩�A�*

epsilonH�B�Q�0.       ��W�	��@̩�A�* 

Average reward per stepH�B���P!       ��2	��@̩�A�*

epsilonH�B��5�U0       ���_	,A̩�A*#
!
Average reward per episode  ���If.       ��W�	A̩�A*!

total reward per episode  
Ö6I.       ��W�	�E̩�A�* 

Average reward per step  ��?H��       ��2	SE̩�A�*

epsilon  ��aЦ.       ��W�	��F̩�A�* 

Average reward per step  ����       ��2	w�F̩�A�*

epsilon  ���6�4.       ��W�	_�H̩�A�* 

Average reward per step  ��K�       ��2	5�H̩�A�*

epsilon  ��Q~�u.       ��W�	7�J̩�A�* 

Average reward per step  ��9Ӄ       ��2	��J̩�A�*

epsilon  �����`.       ��W�	�iM̩�A�* 

Average reward per step  ���,�i       ��2	�jM̩�A�*

epsilon  ��+��.       ��W�	�Q̩�A�* 

Average reward per step  ���f��       ��2	tQ̩�A�*

epsilon  ��k��.       ��W�	�"S̩�A�* 

Average reward per step  ����o       ��2	�#S̩�A�*

epsilon  �����.       ��W�	�:U̩�A�* 

Average reward per step  ����ˎ       ��2	�;U̩�A�*

epsilon  ����ڰ.       ��W�	�EW̩�A�* 

Average reward per step  �����       ��2	�FW̩�A�*

epsilon  ���.       ��W�	�aY̩�A�* 

Average reward per step  ���a�"       ��2	�bY̩�A�*

epsilon  ��޵�.       ��W�	�[̩�A�* 

Average reward per step  ���{5       ��2	��[̩�A�*

epsilon  ����e�.       ��W�	~�\̩�A�* 

Average reward per step  ���W       ��2	�\̩�A�*

epsilon  ��3���.       ��W�	_̩�A�* 

Average reward per step  ��,�]�       ��2	1_̩�A�*

epsilon  ��8%L.       ��W�	�`̩�A�* 

Average reward per step  ��v#�;       ��2	�`̩�A�*

epsilon  ��,s%�.       ��W�	Ԝb̩�A�* 

Average reward per step  ��T��t       ��2	�b̩�A�*

epsilon  ��fF�l.       ��W�	�d̩�A�* 

Average reward per step  ���v�|       ��2	��d̩�A�*

epsilon  ��{�k�.       ��W�	��f̩�A�* 

Average reward per step  ��# q       ��2	��f̩�A�*

epsilon  ���4��.       ��W�	�	i̩�A�* 

Average reward per step  ���b�"       ��2	�
i̩�A�*

epsilon  ���:�;.       ��W�	�|j̩�A�* 

Average reward per step  ��^6�       ��2	�}j̩�A�*

epsilon  ���e��.       ��W�	7Sm̩�A�* 

Average reward per step  ����3�       ��2	�Sm̩�A�*

epsilon  ��C��(.       ��W�	J�n̩�A�* 

Average reward per step  ����        ��2	�n̩�A�*

epsilon  ��	Ws�.       ��W�	F�p̩�A�* 

Average reward per step  �����        ��2	�p̩�A�*

epsilon  ����C�.       ��W�	i�r̩�A�* 

Average reward per step  ��=[�R       ��2	u�r̩�A�*

epsilon  ��?��Q.       ��W�	a�t̩�A�* 

Average reward per step  ��^�       ��2	��t̩�A�*

epsilon  ����Xm.       ��W�	4�w̩�A�* 

Average reward per step  ���C2�       ��2	�w̩�A�*

epsilon  ���5;.       ��W�	-�x̩�A�* 

Average reward per step  ���X       ��2	��x̩�A�*

epsilon  ���w�.       ��W�	��z̩�A�* 

Average reward per step  ����=�       ��2	I�z̩�A�*

epsilon  ���R�!.       ��W�	#�|̩�A�* 

Average reward per step  ����H       ��2	��|̩�A�*

epsilon  ��k�L�.       ��W�	��̩�A�* 

Average reward per step  ��v6�       ��2	��̩�A�*

epsilon  ����<�.       ��W�	8�̩�A�* 

Average reward per step  �� ڑb       ��2	 9�̩�A�*

epsilon  ��:af.       ��W�	�X�̩�A�* 

Average reward per step  ���%�       ��2	uY�̩�A�*

epsilon  ��/ �.       ��W�	젅̩�A�* 

Average reward per step  ��H�       ��2	���̩�A�*

epsilon  ��ixp<.       ��W�	�'�̩�A�* 

Average reward per step  ���O�       ��2	�(�̩�A�*

epsilon  ���k��.       ��W�	���̩�A�* 

Average reward per step  ��Rr��       ��2	���̩�A�*

epsilon  ��u���.       ��W�	R��̩�A�* 

Average reward per step  �����       ��2	0��̩�A�*

epsilon  ����|�.       ��W�	t'�̩�A�* 

Average reward per step  ����R�       ��2	p(�̩�A�*

epsilon  ����X.       ��W�	�N�̩�A�* 

Average reward per step  ���XE       ��2	"O�̩�A�*

epsilon  ��nJLs.       ��W�	�S�̩�A�* 

Average reward per step  ��ńB�       ��2	�T�̩�A�*

epsilon  ���~.       ��W�	~ɓ̩�A�* 

Average reward per step  �����       ��2	Kʓ̩�A�*

epsilon  ��ƚQ�.       ��W�	(ӕ̩�A�* 

Average reward per step  ���fC�       ��2	�ӕ̩�A�*

epsilon  ����+.       ��W�	�9�̩�A�* 

Average reward per step  ��|��       ��2	d:�̩�A�*

epsilon  ����O�.       ��W�	���̩�A�* 

Average reward per step  ����       ��2	���̩�A�*

epsilon  ����9.       ��W�	�e�̩�A�* 

Average reward per step  ���V �       ��2	�f�̩�A�*

epsilon  ���64�.       ��W�	7�̩�A�* 

Average reward per step  ���2�       ��2	 8�̩�A�*

epsilon  �����j.       ��W�	H��̩�A�* 

Average reward per step  ���F|�       ��2	��̩�A�*

epsilon  ��dwLS.       ��W�	.ɣ̩�A�* 

Average reward per step  ��YW�>       ��2	�ɣ̩�A�*

epsilon  ��pRhC.       ��W�	��̩�A�* 

Average reward per step  �����       ��2	��̩�A�*

epsilon  ��*/.       ��W�	�
�̩�A�* 

Average reward per step  ��沝{       ��2	|�̩�A�*

epsilon  ��.S�.       ��W�	�ȩ̩�A�* 

Average reward per step  ����?       ��2	Oɩ̩�A�*

epsilon  ���o��.       ��W�	ȫ̩�A�* 

Average reward per step  ��>�--       ��2	�ȫ̩�A�*

epsilon  ������0       ���_	��̩�A*#
!
Average reward per episode��迠&NW.       ��W�	��̩�A*!

total reward per episode  ��.�8/.       ��W�	ɐ�̩�A�* 

Average reward per step��迯l1       ��2	���̩�A�*

epsilon�����\�.       ��W�	���̩�A�* 

Average reward per step������H       ��2	p��̩�A�*

epsilon���z��m.       ��W�	���̩�A�* 

Average reward per step����ʆa       ��2	���̩�A�*

epsilon��迒j}.       ��W�	Z�̩�A�* 

Average reward per step��迦�Ε       ��2	4��̩�A�*

epsilon���-R�.       ��W�	�$�̩�A�* 

Average reward per step���|(�       ��2	�%�̩�A�*

epsilon��迫�l�.       ��W�	�H�̩�A�* 

Average reward per step��迃;R       ��2	�I�̩�A�*

epsilon���M��a.       ��W�	uU�̩�A�* 

Average reward per step���:�C       ��2	V�̩�A�*

epsilon����P?z.       ��W�	�ѽ̩�A�* 

Average reward per step��迅�+�       ��2	�ҽ̩�A�*

epsilon���R�e�.       ��W�	�9�̩�A�* 

Average reward per step��迃�@       ��2	i:�̩�A�*

epsilon����UJ.       ��W�	=G�̩�A�* 

Average reward per step�����I�       ��2	�G�̩�A�*

epsilon���;ݎ\.       ��W�	���̩�A�* 

Average reward per step�����S       ��2	���̩�A�*

epsilon���n�j.       ��W�	��̩�A�* 

Average reward per step���L���       ��2	c��̩�A�*

epsilon���o�L.       ��W�	��̩�A�* 

Average reward per step��迧��N       ��2	���̩�A�*

epsilon��迷p8.       ��W�	��̩�A�* 

Average reward per step�����C       ��2	��̩�A�*

epsilon�����A.       ��W�	�H�̩�A�* 

Average reward per step���(���       ��2	�I�̩�A�*

epsilon���Ƒ�.       ��W�	���̩�A�* 

Average reward per step��迱$M�       ��2	٘�̩�A�*

epsilon��迥��.       ��W�	���̩�A�* 

Average reward per step��迊���       ��2	���̩�A�*

epsilon���[�6.       ��W�	B@�̩�A�* 

Average reward per step����1r       ��2	5A�̩�A�*

epsilon���^*V.       ��W�	�]�̩�A�* 

Average reward per step���`�4�       ��2	�^�̩�A�*

epsilon���d�.       ��W�	���̩�A�* 

Average reward per step��迦S�3       ��2	\��̩�A�*

epsilon���/�q%.       ��W�	]��̩�A�* 

Average reward per step���Ǌވ       ��2	���̩�A�*

epsilon���XL�C.       ��W�	���̩�A�* 

Average reward per step��迍*�       ��2	���̩�A�*

epsilon���,�
o.       ��W�	?��̩�A�* 

Average reward per step���堰�       ��2	7��̩�A�*

epsilon���%��.       ��W�	5��̩�A�* 

Average reward per step����!�       ��2	��̩�A�*

epsilon����}k0       ���_	j�̩�A*#
!
Average reward per episode  ����c.       ��W�	�̩�A*!

total reward per episode  Ö�]%.       ��W�	#��̩�A�* 

Average reward per step  ��}��f       ��2	��̩�A�*

epsilon  ��a��.       ��W�	���̩�A�* 

Average reward per step  ��;�ߚ       ��2	���̩�A�*

epsilon  ���9N2.       ��W�	3�̩�A�* 

Average reward per step  ���Q       ��2	74�̩�A�*

epsilon  ��XƐS.       ��W�	�R�̩�A�* 

Average reward per step  ���e*       ��2	]S�̩�A�*

epsilon  ���Ȼ�.       ��W�	ٵ�̩�A�* 

Average reward per step  ���>J       ��2	���̩�A�*

epsilon  ������.       ��W�	�G�̩�A�* 

Average reward per step  ���R}�       ��2	RI�̩�A�*

epsilon  ��#�1�.       ��W�	l�̩�A�* 

Average reward per step  ��=H��       ��2	�l�̩�A�*

epsilon  ���^3�.       ��W�	ș�̩�A�* 

Average reward per step  ����(&       ��2	p��̩�A�*

epsilon  ���Y�`.       ��W�	��̩�A�* 

Average reward per step  ��'	 �       ��2	��̩�A�*

epsilon  ���ە.       ��W�	���̩�A�* 

Average reward per step  ���7�>       ��2	1��̩�A�*

epsilon  ����9(.       ��W�	�A�̩�A�* 

Average reward per step  ���"�K       ��2	9B�̩�A�*

epsilon  ��T��j.       ��W�	�=�̩�A�* 

Average reward per step  ��$�hh       ��2	�>�̩�A�*

epsilon  ��x@�.       ��W�	{��̩�A�* 

Average reward per step  ��_W�v       ��2	Y��̩�A�*

epsilon  ��;i.       ��W�	�@�̩�A�* 

Average reward per step  ��6�1r       ��2	hA�̩�A�*

epsilon  �����.       ��W�	���̩�A�* 

Average reward per step  ���J>       ��2	2��̩�A�*

epsilon  ����*c.       ��W�	c)̩�A�* 

Average reward per step  ��&aE       ��2	A*̩�A�*

epsilon  �����}.       ��W�	9�̩�A�* 

Average reward per step  ��\)�&       ��2	�̩�A�*

epsilon  ��R�)�.       ��W�	Ҭ̩�A�* 

Average reward per step  ����A`       ��2	i�̩�A�*

epsilon  �����Z.       ��W�	*�̩�A�* 

Average reward per step  ��m'<�       ��2	�̩�A�*

epsilon  ���݇>.       ��W�	'�̩�A�* 

Average reward per step  ���HL;       ��2	�̩�A�*

epsilon  ���9��0       ���_	;�̩�A*#
!
Average reward per episode  ���h8.       ��W�	��̩�A*!

total reward per episode  %Â�[�.       ��W�	��̩�A�* 

Average reward per step  �_E       ��2	��̩�A�*

epsilon  �\���.       ��W�	��̩�A�* 

Average reward per step  ��<       ��2	e�̩�A�*

epsilon  � �a.       ��W�	��̩�A�* 

Average reward per step  �d�^H       ��2	v�̩�A�*

epsilon  ���N�.       ��W�	#�̩�A�* 

Average reward per step  ��@f       ��2	��̩�A�*

epsilon  ���S.       ��W�	W	̩�A�* 

Average reward per step  ��Ɦ       ��2	=̩�A�*

epsilon  �w=�.       ��W�	��̩�A�* 

Average reward per step  �j��       ��2	p�̩�A�*

epsilon  ���]J.       ��W�	L�̩�A�* 

Average reward per step  �����       ��2	&�̩�A�*

epsilon  ��-��.       ��W�	��̩�A�* 

Average reward per step  ���}       ��2	��̩�A�*

epsilon  �V�"�.       ��W�	0�̩�A�* 

Average reward per step  �g�L       ��2	�̩�A�*

epsilon  �L�lJ.       ��W�	\ ̩�A�* 

Average reward per step  �;B��       ��2	!̩�A�*

epsilon  ����.       ��W�	& ̩�A�* 

Average reward per step  �|��       ��2	� ̩�A�*

epsilon  �\#y�.       ��W�	K�"̩�A�* 

Average reward per step  �8(�       ��2	!�"̩�A�*

epsilon  �'"X�.       ��W�	�V&̩�A�* 

Average reward per step  �B���       ��2	�W&̩�A�*

epsilon  ��3��.       ��W�	"�(̩�A�* 

Average reward per step  �Z�U�       ��2	�(̩�A�*

epsilon  �{�]�.       ��W�	 ^*̩�A�* 

Average reward per step  �����       ��2	x_*̩�A�*

epsilon  ��b��.       ��W�	a�,̩�A�* 

Average reward per step  ���$L       ��2	��,̩�A�*

epsilon  ��=�:.       ��W�	�f.̩�A�* 

Average reward per step  ���]       ��2	�g.̩�A�*

epsilon  �q��;.       ��W�	��0̩�A�* 

Average reward per step  �FWy�       ��2	4�0̩�A�*

epsilon  �7�a�0       ���_	غ0̩�A*#
!
Average reward per episode������.       ��W�	^�0̩�A*!

total reward per episode  #�w��b.       ��W�	k�4̩�A�* 

Average reward per step�����C�       ��2	^�4̩�A�*

epsilon�����{�.       ��W�	p�6̩�A�* 

Average reward per step���N)n�       ��2	:�6̩�A�*

epsilon���=y�..       ��W�	�8̩�A�* 

Average reward per step����~�       ��2	��8̩�A�*

epsilon���0�>�.       ��W�	��:̩�A�* 

Average reward per step���g�       ��2	��:̩�A�*

epsilon���<F�.       ��W�	�<̩�A�* 

Average reward per step���+��x       ��2	��<̩�A�*

epsilon������.       ��W�	�1?̩�A�* 

Average reward per step����S5�       ��2	r2?̩�A�*

epsilon���wp'c.       ��W�	�@̩�A�* 

Average reward per step���_�j       ��2	u�@̩�A�*

epsilon����x�.       ��W�	��B̩�A�* 

Average reward per step����lnG       ��2	��B̩�A�*

epsilon���)_x4.       ��W�	x�D̩�A�* 

Average reward per step���Z�       ��2	E�D̩�A�*

epsilon�����PJ.       ��W�	��F̩�A�* 

Average reward per step���#h�       ��2	z�F̩�A�*

epsilon�����.       ��W�	�	I̩�A�* 

Average reward per step�����p�       ��2	g
I̩�A�*

epsilon���G^�.       ��W�	��J̩�A�* 

Average reward per step����g��       ��2	}�J̩�A�*

epsilon���N�B�.       ��W�	-\M̩�A�* 

Average reward per step���:h��       ��2	]M̩�A�*

epsilon���e�{�.       ��W�	I�O̩�A�* 

Average reward per step����ud       ��2	�O̩�A�*

epsilon����*��.       ��W�	�Q̩�A�* 

Average reward per step���)=h�       ��2	�Q̩�A�*

epsilon���1�..       ��W�	fS̩�A�* 

Average reward per step���
d�       ��2	+S̩�A�*

epsilon����G	�.       ��W�	�U̩�A�* 

Average reward per step���Ur-�       ��2	�U̩�A�*

epsilon����;F�.       ��W�	�UW̩�A�* 

Average reward per step������       ��2	�WW̩�A�*

epsilon����e.       ��W�	��Y̩�A�* 

Average reward per step���Rօ'       ��2	��Y̩�A�*

epsilon���	��.       ��W�	��[̩�A�* 

Average reward per step���ٖ�       ��2	��[̩�A�*

epsilon���]n��.       ��W�	��]̩�A�* 

Average reward per step���.q�       ��2	��]̩�A�*

epsilon����b�D.       ��W�	;S_̩�A�* 

Average reward per step������       ��2	�S_̩�A�*

epsilon����X�D.       ��W�	�Pa̩�A�* 

Average reward per step���w���       ��2	;Qa̩�A�*

epsilon�����.       ��W�	�wc̩�A�* 

Average reward per step�������       ��2	`xc̩�A�*

epsilon���ὼ�.       ��W�	�f̩�A�* 

Average reward per step���:��       ��2	�f̩�A�*

epsilon���З+1.       ��W�	3�g̩�A�* 

Average reward per step���Y�ox       ��2	L�g̩�A�*

epsilon���Q��.       ��W�	��i̩�A�* 

Average reward per step��� e��       ��2	)�i̩�A�*

epsilon���Lw�&.       ��W�	�Bk̩�A�* 

Average reward per step���euL�       ��2	gCk̩�A�*

epsilon�����.e.       ��W�	n�l̩�A�* 

Average reward per step����b�	       ��2	�l̩�A�*

epsilon������.       ��W�	S�n̩�A�* 

Average reward per step���޽�       ��2	%�n̩�A�*

epsilon���f�A�.       ��W�	z�p̩�A�* 

Average reward per step����sX       ��2	n�p̩�A�*

epsilon���s��.       ��W�	p�r̩�A�* 

Average reward per step���*�;�       ��2	B�r̩�A�*

epsilon���9�R+.       ��W�	'�t̩�A�* 

Average reward per step���M��       ��2	�t̩�A�*

epsilon����U�.       ��W�	1^v̩�A�* 

Average reward per step���t9&�       ��2	g_v̩�A�*

epsilon���)���.       ��W�	��x̩�A�* 

Average reward per step����PSk       ��2	F�x̩�A�*

epsilon���8Γ�.       ��W�	V�{̩�A�* 

Average reward per step���-1�       ��2	�{̩�A�*

epsilon���csA�.       ��W�	O̩�A�* 

Average reward per step����gP�       ��2	̩�A�*

epsilon���A/ġ.       ��W�	A�̩�A�* 

Average reward per step����_�b       ��2	#�̩�A�*

epsilon�����.       ��W�	�$�̩�A�* 

Average reward per step�����1       ��2	[%�̩�A�*

epsilon���R|�b.       ��W�	\X�̩�A�* 

Average reward per step���;R       ��2	mY�̩�A�*

epsilon���7ld.       ��W�	|�̩�A�* 

Average reward per step���C]�8       ��2	�|�̩�A�*

epsilon�����.       ��W�	
��̩�A�* 

Average reward per step����ԕ"       ��2	���̩�A�*

epsilon���q�H.       ��W�	߉�̩�A�* 

Average reward per step�����o       ��2	���̩�A�*

epsilon����RP.       ��W�	��̩�A�* 

Average reward per step���a�z-       ��2	ß�̩�A�*

epsilon���8xgR.       ��W�	^�̩�A�* 

Average reward per step�����       ��2	<�̩�A�*

epsilon�����_X.       ��W�	���̩�A�* 

Average reward per step����I_       ��2	���̩�A�*

epsilon���e/�.       ��W�	`��̩�A�* 

Average reward per step���c��       ��2	 ��̩�A�*

epsilon����cM.       ��W�	���̩�A�* 

Average reward per step�����&       ��2	���̩�A�*

epsilon������h.       ��W�	�֗̩�A�* 

Average reward per step���Pc       ��2	�ח̩�A�*

epsilon���]�.       ��W�	n5�̩�A�* 

Average reward per step����X��       ��2	P6�̩�A�*

epsilon�����
J.       ��W�	y�̩�A�* 

Average reward per step���³�1       ��2	l�̩�A�*

epsilon�����nN.       ��W�	*R�̩�A�* 

Average reward per step�������       ��2	�R�̩�A�*

epsilon���Rm[�.       ��W�	�y�̩�A�* 

Average reward per step�����{R       ��2	�z�̩�A�*

epsilon���n��i.       ��W�	�ߡ̩�A�* 

Average reward per step�����D�       ��2	��̩�A�*

epsilon�����.       ��W�	�!�̩�A�* 

Average reward per step���5��       ��2	�#�̩�A�*

epsilon���[��.       ��W�	σ�̩�A�* 

Average reward per step���v�9       ��2	���̩�A�*

epsilon�������.       ��W�	�J�̩�A�* 

Average reward per step����2��       ��2	{K�̩�A�*

epsilon���aC)�.       ��W�	h�̩�A�* 

Average reward per step����|�*       ��2	��̩�A�*

epsilon���m29..       ��W�	� �̩�A�* 

Average reward per step���M[�       ��2	S!�̩�A�*

epsilon���`�.       ��W�	�Z�̩�A�* 

Average reward per step���v_[�       ��2	p[�̩�A�*

epsilon���~_&.       ��W�	E�̩�A�* 

Average reward per step����l�       ��2	w�̩�A�*

epsilon���E��.       ��W�	�1�̩�A�* 

Average reward per step����FN�       ��2	�2�̩�A�*

epsilon�������.       ��W�	y��̩�A�* 

Average reward per step�������       ��2	[��̩�A�*

epsilon����C�e.       ��W�	qV�̩�A�* 

Average reward per step�����G       ��2	iW�̩�A�*

epsilon����]�.       ��W�	债̩�A�* 

Average reward per step���I��)       ��2	���̩�A�*

epsilon����h�.       ��W�	_�̩�A�* 

Average reward per step����g       ��2	��̩�A�*

epsilon������..       ��W�	伾̩�A�* 

Average reward per step���z�       ��2	ӽ�̩�A�*

epsilon����2�c.       ��W�	�K�̩�A�* 

Average reward per step����8��       ��2	M�̩�A�*

epsilon���'	X.       ��W�	�Q�̩�A�* 

Average reward per step���T���       ��2	�R�̩�A�*

epsilon���m��.       ��W�	���̩�A�* 

Average reward per step����cb�       ��2	4��̩�A�*

epsilon����c�.       ��W�	���̩�A�* 

Average reward per step������;       ��2	E��̩�A�*

epsilon���TF��.       ��W�	���̩�A�* 

Average reward per step����/�       ��2	���̩�A�*

epsilon����;K.       ��W�	1��̩�A�* 

Average reward per step����hE�       ��2	���̩�A�*

epsilon����v��0       ���_	��̩�A*#
!
Average reward per episode#F�6�99.       ��W�	���̩�A*!

total reward per episode   ���.       ��W�	%��̩�A�* 

Average reward per step#F�����       ��2	��̩�A�*

epsilon#F�X�.       ��W�	���̩�A�* 

Average reward per step#F�u��l       ��2	���̩�A�*

epsilon#F��.       ��W�	�=�̩�A�* 

Average reward per step#F�m��D       ��2	�>�̩�A�*

epsilon#F��2��.       ��W�	���̩�A�* 

Average reward per step#F�����       ��2	���̩�A�*

epsilon#F�AMt.       ��W�	>��̩�A�* 

Average reward per step#F����C       ��2	F��̩�A�*

epsilon#F�����.       ��W�	G��̩�A�* 

Average reward per step#F���!       ��2	!��̩�A�*

epsilon#F�$��:.       ��W�	���̩�A�* 

Average reward per step#F�6qy       ��2	���̩�A�*

epsilon#F��g)�.       ��W�	�̩�A�* 

Average reward per step#F�NȦ�       ��2	�̩�A�*

epsilon#F�����.       ��W�	���̩�A�* 

Average reward per step#F��lr�       ��2	���̩�A�*

epsilon#F�_G{�.       ��W�	>�̩�A�* 

Average reward per step#F�c�r�       ��2	�>�̩�A�*

epsilon#F�GF_�.       ��W�	&��̩�A�* 

Average reward per step#F���Tv       ��2	H��̩�A�*

epsilon#F�ȷ-�.       ��W�	n��̩�A�* 

Average reward per step#F��|�       ��2	;��̩�A�*

epsilon#F��h.       ��W�	���̩�A�* 

Average reward per step#F�7(�I       ��2	��̩�A�*

epsilon#F�*�3f.       ��W�	�0�̩�A�* 

Average reward per step#F���G#       ��2	a3�̩�A�*

epsilon#F�Z�.       ��W�	��̩�A�* 

Average reward per step#F�]%�V       ��2	��̩�A�*

epsilon#F�:�:�.       ��W�	���̩�A�* 

Average reward per step#F�
1d       ��2	��̩�A�*

epsilon#F��Q4 .       ��W�	�I�̩�A�* 

Average reward per step#F���       ��2	4K�̩�A�*

epsilon#F����.       ��W�	���̩�A�* 

Average reward per step#F���ڦ       ��2	���̩�A�*

epsilon#F���'.       ��W�	s��̩�A�* 

Average reward per step#F�H�w�       ��2	Z��̩�A�*

epsilon#F�� �8.       ��W�	9|�̩�A�* 

Average reward per step#F�&��
       ��2	�~�̩�A�*

epsilon#F��<V�.       ��W�	#��̩�A�* 

Average reward per step#F�#��       ��2	��̩�A�*

epsilon#F�]��T.       ��W�	K;�̩�A�* 

Average reward per step#F��b��       ��2	%<�̩�A�*

epsilon#F�x��.       ��W�	���̩�A�* 

Average reward per step#F�@�       ��2	���̩�A�*

epsilon#F���0       ���_	�=�̩�A*#
!
Average reward per episodez��� ��.       ��W�	�>�̩�A*!

total reward per episode  ��2�.       ��W�	yv̩�A�* 

Average reward per stepz���<���       ��2	�w̩�A�*

epsilonz���U︂.       ��W�	 (	̩�A�* 

Average reward per stepz����5�Q       ��2	V)	̩�A�*

epsilonz�����.       ��W�	�7̩�A�* 

Average reward per stepz�������       ��2	�8̩�A�*

epsilonz�����<p.       ��W�	(̩�A�* 

Average reward per stepz���	>��       ��2	()̩�A�*

epsilonz���|���.       ��W�	�`̩�A�* 

Average reward per stepz�����(L       ��2	b̩�A�*

epsilonz�����o.       ��W�	�.̩�A�* 

Average reward per stepz�����<z       ��2	�/̩�A�*

epsilonz���D�.       ��W�	V�̩�A�* 

Average reward per stepz���W$�       ��2	��̩�A�*

epsilonz���D��.       ��W�	M�̩�A�* 

Average reward per stepz���ҍ�       ��2	{�̩�A�*

epsilonz����7��.       ��W�	37̩�A�* 

Average reward per stepz�����Dx       ��2		8̩�A�*

epsilonz����u�).       ��W�	ܻ̩�A�* 

Average reward per stepz���~5f       ��2	��̩�A�*

epsilonz�����N.       ��W�	�J̩�A�* 

Average reward per stepz���\�:�       ��2	�K̩�A�*

epsilonz���-T?�.       ��W�	q�!̩�A�* 

Average reward per stepz����hL�       ��2	`�!̩�A�*

epsilonz���{��.       ��W�	��#̩�A�* 

Average reward per stepz���6���       ��2	��#̩�A�*

epsilonz���-o�.       ��W�	q�%̩�A�* 

Average reward per stepz���&M^�       ��2	d�%̩�A�*

epsilonz���a9��.       ��W�	(̩�A�* 

Average reward per stepz��� �       ��2	Y(̩�A�*

epsilonz���puP.       ��W�	�)̩�A�* 

Average reward per stepz������       ��2	��)̩�A�*

epsilonz����J.       ��W�	� ,̩�A�* 

Average reward per stepz���>ơ&       ��2	�!,̩�A�*

epsilonz���*Ĝ>.       ��W�	��-̩�A�* 

Average reward per stepz����ŕH       ��2	��-̩�A�*

epsilonz�����8.       ��W�	0̩�A�* 

Average reward per stepz����Q�       ��2	20̩�A�*

epsilonz���-X$q.       ��W�	�r2̩�A�* 

Average reward per stepz�����Y�       ��2	�s2̩�A�*

epsilonz���;��.       ��W�	V(6̩�A�* 

Average reward per stepz���o߭�       ��2	�)6̩�A�*

epsilonz�����?w.       ��W�	Ț8̩�A�* 

Average reward per stepz���ʋ�l       ��2	̛8̩�A�*

epsilonz���|0P\.       ��W�	?;̩�A�* 

Average reward per stepz���jX��       ��2	*;̩�A�*

epsilonz����yaO.       ��W�	؝>̩�A�* 

Average reward per stepz���U��       ��2	��>̩�A�*

epsilonz������.       ��W�	V@̩�A�* 

Average reward per stepz���J��       ��2	&W@̩�A�*

epsilonz���:�o�.       ��W�	��B̩�A�* 

Average reward per stepz�����       ��2	ɎB̩�A�*

epsilonz����2�.       ��W�		�D̩�A�* 

Average reward per stepz������       ��2	.�D̩�A�*

epsilonz�����.       ��W�	�G̩�A�* 

Average reward per stepz���\e��       ��2	wG̩�A�*

epsilonz���"K>i.       ��W�	ϡJ̩�A�* 

Average reward per stepz������[       ��2	�J̩�A�*

epsilonz���]yC.       ��W�	9dL̩�A�* 

Average reward per stepz����w��       ��2	weL̩�A�*

epsilonz�����n�.       ��W�	��N̩�A�* 

Average reward per stepz������Y       ��2	�N̩�A�*

epsilonz����z��.       ��W�	;Q̩�A�* 

Average reward per stepz���!�       ��2	]Q̩�A�*

epsilonz�������.       ��W�	��R̩�A�* 

Average reward per stepz���U���       ��2	��R̩�A�*

epsilonz������@.       ��W�	_$U̩�A�* 

Average reward per stepz����/I�       ��2	&U̩�A�*

epsilonz���e��.       ��W�	�W̩�A�* 

Average reward per stepz���:�        ��2	/�W̩�A�*

epsilonz�����xq0       ���_	�W̩�A*#
!
Average reward per episode�$I�>W�.       ��W�	7�W̩�A*!

total reward per episode  ��١�.       ��W�	�T]̩�A�* 

Average reward per step�$I�[zS�       ��2	�U]̩�A�*

epsilon�$I��b�'.       ��W�		_̩�A�* 

Average reward per step�$I� \'�       ��2	�	_̩�A�*

epsilon�$I�~��r.       ��W�	��a̩�A�* 

Average reward per step�$I�y�:       ��2	�a̩�A�*

epsilon�$I��H��.       ��W�	4c̩�A�* 

Average reward per step�$I��lF�       ��2	c̩�A�*

epsilon�$I��o��.       ��W�	'le̩�A�* 

Average reward per step�$I�ck"       ��2	ame̩�A�*

epsilon�$I�����.       ��W�	��g̩�A�* 

Average reward per step�$I�7[8       ��2	ƾg̩�A�*

epsilon�$I��!�3.       ��W�	zPi̩�A�* 

Average reward per step�$I�z�8<       ��2	�Qi̩�A�*

epsilon�$I�<��.       ��W�	�k̩�A�* 

Average reward per step�$I�*���       ��2	�k̩�A�*

epsilon�$I��(�.       ��W�	Vm̩�A�* 

Average reward per step�$I��L8Z       ��2	KWm̩�A�*

epsilon�$I��y�|.       ��W�	��o̩�A�* 

Average reward per step�$I�h�        ��2	�o̩�A�*

epsilon�$I�OU+.       ��W�	�q̩�A�* 

Average reward per step�$I�Bטq       ��2	Q�q̩�A�*

epsilon�$I�^O .       ��W�	Às̩�A�* 

Average reward per step�$I����       ��2	��s̩�A�*

epsilon�$I��,kT.       ��W�	��u̩�A�* 

Average reward per step�$I��88s       ��2	�u̩�A�*

epsilon�$I�����.       ��W�	?�w̩�A�* 

Average reward per step�$I��       ��2	�w̩�A�*

epsilon�$I�鿧�.       ��W�	 Bz̩�A�* 

Average reward per step�$I��>��       ��2	�Bz̩�A�*

epsilon�$I�.>��.       ��W�	��{̩�A�* 

Average reward per step�$I���A�       ��2	��{̩�A�*

epsilon�$I�η.       ��W�	i�}̩�A�* 

Average reward per step�$I�!���       ��2	?�}̩�A�*

epsilon�$I���zO.       ��W�	ސ�̩�A�* 

Average reward per step�$I��.�       ��2	��̩�A�*

epsilon�$I��gy.       ��W�	[A�̩�A�* 

Average reward per step�$I�ڹ?�       ��2	1B�̩�A�*

epsilon�$I�=܋n0       ���_	�d�̩�A*#
!
Average reward per episode�������.       ��W�	�e�̩�A*!

total reward per episode  �x��.       ��W�	�E�̩�A�* 

Average reward per step�����J�4       ��2	|F�̩�A�*

epsilon�����T.       ��W�	�̩�A�* 

Average reward per step����\�9�       ��2	��̩�A�*

epsilon����q?�&.       ��W�	&:�̩�A�* 

Average reward per step�����F�       ��2	;�̩�A�*

epsilon�����2.       ��W�	�Ҏ̩�A�* 

Average reward per step�����e+�       ��2	�ӎ̩�A�*

epsilon����vg.       ��W�	>�̩�A�* 

Average reward per step����ݵ��       ��2	�>�̩�A�*

epsilon�������g.       ��W�	�D�̩�A�* 

Average reward per step����Q���       ��2	�E�̩�A�*

epsilon�����98�.       ��W�	¡�̩�A�* 

Average reward per step�����pfX       ��2	��̩�A�*

epsilon�������.       ��W�	���̩�A�* 

Average reward per step����c$>�       ��2	陜̩�A�*

epsilon�����y�a.       ��W�	���̩�A�* 

Average reward per step������@�       ��2	���̩�A�*

epsilon����,��:.       ��W�	�E�̩�A�* 

Average reward per step�������       ��2	�F�̩�A�*

epsilon����[?��.       ��W�	�̩�A�* 

Average reward per step������u       ��2	2�̩�A�*

epsilon����z�wD.       ��W�	>[�̩�A�* 

Average reward per step����L3�       ��2	�\�̩�A�*

epsilon����^�p.       ��W�	��̩�A�* 

Average reward per step�����<�       ��2	��̩�A�*

epsilon����T���.       ��W�	�[�̩�A�* 

Average reward per step������F�       ��2	�\�̩�A�*

epsilon����#��j.       ��W�	8��̩�A�* 

Average reward per step����7F       ��2	H��̩�A�*

epsilon����:��.       ��W�	:�̩�A�* 

Average reward per step����i��       ��2	C;�̩�A�*

epsilon������F.       ��W�	�D�̩�A�* 

Average reward per step�������:       ��2	�E�̩�A�*

epsilon�������`.       ��W�	��̩�A�* 

Average reward per step������E       ��2	��̩�A�*

epsilon����U.       ��W�	~��̩�A�* 

Average reward per step����nEM       ��2	֍�̩�A�*

epsilon������;�.       ��W�	�C�̩�A�* 

Average reward per step������f       ��2	�D�̩�A�*

epsilon����)��.       ��W�	���̩�A�* 

Average reward per step������       ��2	ꓽ̩�A�*

epsilon����eװ.       ��W�	nP�̩�A�* 

Average reward per step�����<�C       ��2	�Q�̩�A�*

epsilon����i�g�.       ��W�	�̩�A�* 

Average reward per step������%       ��2	�̩�A�*

epsilon����#.       ��W�	��̩�A�* 

Average reward per step����M       ��2	��̩�A�*

epsilon�����}�Q.       ��W�	���̩�A�* 

Average reward per step�������W       ��2	���̩�A�*

epsilon����^��.       ��W�	q�̩�A�* 

Average reward per step��������       ��2	r�̩�A�*

epsilon�����]B0       ���_	���̩�A*#
!
Average reward per episode  ��@��.       ��W�	_��̩�A*!

total reward per episode  Í"Z�.       ��W�	�Q�̩�A�* 

Average reward per step  ����\a       ��2	�R�̩�A�*

epsilon  ��ۅ/.       ��W�	I��̩�A�* 

Average reward per step  ��SC�       ��2	���̩�A�*

epsilon  ��Į=>.       ��W�	���̩�A�* 

Average reward per step  ��yِ�       ��2	q��̩�A�*

epsilon  ��% �~.       ��W�	���̩�A�* 

Average reward per step  ��[�t       ��2	���̩�A�*

epsilon  ���#�.       ��W�	�1�̩�A�* 

Average reward per step  ����X       ��2	j2�̩�A�*

epsilon  ������.       ��W�	?6�̩�A�* 

Average reward per step  ��^���       ��2	�6�̩�A�*

epsilon  ��T��.       ��W�	JC�̩�A�* 

Average reward per step  ���	�       ��2	�C�̩�A�*

epsilon  ��t��(.       ��W�	���̩�A�* 

Average reward per step  ��-���       ��2	���̩�A�*

epsilon  ��6�;J.       ��W�	i��̩�A�* 

Average reward per step  ��&+(       ��2	���̩�A�*

epsilon  ��|��.       ��W�	O"�̩�A�* 

Average reward per step  ��܇�p       ��2	-#�̩�A�*

epsilon  ���!��.       ��W�	w��̩�A�* 

Average reward per step  ��Yz�p       ��2	��̩�A�*

epsilon  ���`�.       ��W�	��̩�A�* 

Average reward per step  ����*       ��2	���̩�A�*

epsilon  ����I.       ��W�	��̩�A�* 

Average reward per step  ������       ��2	ő�̩�A�*

epsilon  ���A��.       ��W�	���̩�A�* 

Average reward per step  ��!��v       ��2	g��̩�A�*

epsilon  ��I
.�.       ��W�	��̩�A�* 

Average reward per step  ��2��       ��2	���̩�A�*

epsilon  ��	E�.       ��W�	���̩�A�* 

Average reward per step  ������       ��2	���̩�A�*

epsilon  ���.       ��W�	��̩�A�* 

Average reward per step  ���7�       ��2	���̩�A�*

epsilon  �����.       ��W�	�O�̩�A�* 

Average reward per step  ���dC[       ��2	jP�̩�A�*

epsilon  ��AMJ.       ��W�	(��̩�A�* 

Average reward per step  ��zLؓ       ��2	���̩�A�*

epsilon  ��`�$.       ��W�	a��̩�A�* 

Average reward per step  ��(w��       ��2	 ��̩�A�*

epsilon  ���F�.       ��W�	�.̩�A�* 

Average reward per step  ��If7�       ��2	�/̩�A�*

epsilon  ��]�s0       ���_	�V̩�A*#
!
Average reward per episodez����K_.       ��W�	OW̩�A*!

total reward per episode  �jg:�.       ��W�	�7̩�A�* 

Average reward per stepz�������       ��2	i8̩�A�*

epsilonz����l�.       ��W�	"�̩�A�* 

Average reward per stepz�����ҫ       ��2	��̩�A�*

epsilonz���N�MM.       ��W�	8�
̩�A�* 

Average reward per stepz����~�       ��2	��
̩�A�*

epsilonz����.       ��W�	��̩�A�* 

Average reward per stepz���('��       ��2	z ̩�A�*

epsilonz����X�q.       ��W�	{̩�A�* 

Average reward per stepz�������       ��2	8̩�A�*

epsilonz����ݏ�.       ��W�	z4̩�A�* 

Average reward per stepz����]U       ��2	L5̩�A�*

epsilonz���!���.       ��W�	�?̩�A�* 

Average reward per stepz���v�&       ��2	�@̩�A�*

epsilonz����*@.       ��W�	(�̩�A�* 

Average reward per stepz����R�7       ��2	�̩�A�*

epsilonz���u�Y�.       ��W�	6;̩�A�* 

Average reward per stepz����b�       ��2	�;̩�A�*

epsilonz����`L.       ��W�	�L̩�A�* 

Average reward per stepz������C       ��2	rM̩�A�*

epsilonz����A�.       ��W�	�x̩�A�* 

Average reward per stepz���l{�       ��2	�y̩�A�*

epsilonz���g|H..       ��W�	�̩�A�* 

Average reward per stepz�������       ��2	��̩�A�*

epsilonz���-��.       ��W�	��̩�A�* 

Average reward per stepz���X�S>       ��2	��̩�A�*

epsilonz�����J�.       ��W�	4G!̩�A�* 

Average reward per stepz����}(�       ��2	H!̩�A�*

epsilonz���T'Ǽ.       ��W�	Gu#̩�A�* 

Average reward per stepz���?p��       ��2	�u#̩�A�*

epsilonz���y�.       ��W�	��%̩�A�* 

Average reward per stepz����[�|       ��2	��%̩�A�*

epsilonz����pi3.       ��W�	��'̩�A�* 

Average reward per stepz����R       ��2	�'̩�A�*

epsilonz����p�.       ��W�	<*̩�A�* 

Average reward per stepz���}��N       ��2	�*̩�A�*

epsilonz�����j3.       ��W�	cA,̩�A�* 

Average reward per stepz����%��       ��2	B,̩�A�*

epsilonz����7So.       ��W�	j�-̩�A�* 

Average reward per stepz�����S�       ��2	�-̩�A�*

epsilonz���
$t.       ��W�	��/̩�A�* 

Average reward per stepz���ɉv@       ��2	O�/̩�A�*

epsilonz���E��.       ��W�	��1̩�A�* 

Average reward per stepz����]"�       ��2	4�1̩�A�*

epsilonz����Ѱ.       ��W�	Z4̩�A�* 

Average reward per stepz����ڢ       ��2	�Z4̩�A�*

epsilonz����{Q�.       ��W�	�5̩�A�* 

Average reward per stepz����W�R       ��2	9�5̩�A�*

epsilonz���4�)�.       ��W�	��7̩�A�* 

Average reward per stepz�����x_       ��2	��7̩�A�*

epsilonz������.       ��W�	�:̩�A�* 

Average reward per stepz���l�       ��2	[	:̩�A�*

epsilonz�������.       ��W�	�<̩�A�* 

Average reward per stepz����X�       ��2	*<̩�A�*

epsilonz����F^.       ��W�	�`>̩�A�* 

Average reward per stepz�����q[       ��2	Va>̩�A�*

epsilonz���ߗW�.       ��W�	�@̩�A�* 

Average reward per stepz�����m       ��2	��@̩�A�*

epsilonz�������.       ��W�	ƥB̩�A�* 

Average reward per stepz����X       ��2	e�B̩�A�*

epsilonz���nۈ�.       ��W�	7�D̩�A�* 

Average reward per stepz���zY       ��2	�D̩�A�*

epsilonz�����Ib.       ��W�	�F̩�A�* 

Average reward per stepz���8�V�       ��2	��F̩�A�*

epsilonz���*� =.       ��W�	�H̩�A�* 

Average reward per stepz����׫l       ��2	�H̩�A�*

epsilonz���G�Na.       ��W�	��J̩�A�* 

Average reward per stepz���byl       ��2	��J̩�A�*

epsilonz���r��].       ��W�	�WL̩�A�* 

Average reward per stepz�������       ��2	}XL̩�A�*

epsilonz���TC.       ��W�	�ZN̩�A�* 

Average reward per stepz����� "       ��2	W[N̩�A�*

epsilonz�����.       ��W�	�nP̩�A�* 

Average reward per stepz���<�5�       ��2	eoP̩�A�*

epsilonz���[�d�.       ��W�	@MS̩�A�* 

Average reward per stepz�����|       ��2	MNS̩�A�*

epsilonz���ϻ0�.       ��W�	��V̩�A�* 

Average reward per stepz������       ��2	��V̩�A�*

epsilonz���8o�.       ��W�	;�X̩�A�* 

Average reward per stepz����j��       ��2	�X̩�A�*

epsilonz���R��.       ��W�	�#[̩�A�* 

Average reward per stepz���n�F�       ��2	B$[̩�A�*

epsilonz����y2�.       ��W�	�[]̩�A�* 

Average reward per stepz�������       ��2	�\]̩�A�*

epsilonz����+�n.       ��W�	�_̩�A�* 

Average reward per stepz����j�       ��2	��_̩�A�*

epsilonz���N+�T.       ��W�	�a̩�A�* 

Average reward per stepz�����-       ��2	�a̩�A�*

epsilonz����آZ.       ��W�	c̩�A�* 

Average reward per stepz���`�E�       ��2	�c̩�A�*

epsilonz���Ձ�.       ��W�	g*e̩�A�* 

Average reward per stepz���� �x       ��2	A+e̩�A�*

epsilonz������.       ��W�	,Gg̩�A�* 

Average reward per stepz���W6:�       ��2	�Gg̩�A�*

epsilonz���� m�.       ��W�	aTi̩�A�* 

Average reward per stepz���w��}       ��2	 Ui̩�A�*

epsilonz����&��.       ��W�	bk̩�A�* 

Average reward per stepz����z�       ��2	�bk̩�A�*

epsilonz���M�g.       ��W�	�pm̩�A�* 

Average reward per stepz����D��       ��2	\qm̩�A�*

epsilonz����?.       ��W�	�o̩�A�* 

Average reward per stepz���N�u�       ��2	ۋo̩�A�*

epsilonz�����׃.       ��W�	M�q̩�A�* 

Average reward per stepz���&g�       ��2	+�q̩�A�*

epsilonz����U5.       ��W�	��s̩�A�* 

Average reward per stepz�������       ��2	/�s̩�A�*

epsilonz���i�X.       ��W�	 �u̩�A�* 

Average reward per stepz��� �m       ��2	��u̩�A�*

epsilonz���ߑ��.       ��W�	�[w̩�A�* 

Average reward per stepz����{��       ��2	�\w̩�A�*

epsilonz�����m.       ��W�	�ty̩�A�* 

Average reward per stepz���^G�       ��2	�uy̩�A�*

epsilonz������.       ��W�	�}{̩�A�* 

Average reward per stepz����2�e       ��2	1~{̩�A�*

epsilonz���m�
2.       ��W�	׉}̩�A�* 

Average reward per stepz���#��       ��2	r�}̩�A�*

epsilonz�����(�.       ��W�	˝̩�A�* 

Average reward per stepz���1ŭ       ��2	Ǟ̩�A�*

epsilonz�������.       ��W�	�ā̩�A�* 

Average reward per stepz������=       ��2	�Ł̩�A�*

epsilonz����	R�.       ��W�	�B�̩�A�* 

Average reward per stepz����g��       ��2	JC�̩�A�*

epsilonz����ê�.       ��W�	�Մ̩�A�* 

Average reward per stepz�������       ��2	kք̩�A�*

epsilonz����l�'.       ��W�	���̩�A�* 

Average reward per stepz������       ��2	噆̩�A�*

epsilonz���Bҋe.       ��W�	�ň̩�A�* 

Average reward per stepz���
���       ��2	�ƈ̩�A�*

epsilonz����qx�.       ��W�	��̩�A�* 

Average reward per stepz�����u       ��2	>�̩�A�*

epsilonz������.       ��W�	��̩�A�* 

Average reward per stepz���a-�j       ��2	��̩�A�*

epsilonz����8��.       ��W�	p"�̩�A�* 

Average reward per stepz����v��       ��2	#�̩�A�*

epsilonz�����WY.       ��W�	�9�̩�A�* 

Average reward per stepz���~�       ��2	�:�̩�A�*

epsilonz������.       ��W�	���̩�A�* 

Average reward per stepz���5���       ��2	h��̩�A�*

epsilonz����P�+.       ��W�	÷�̩�A�* 

Average reward per stepz���=E`�       ��2	���̩�A�*

epsilonz����#��.       ��W�	�k�̩�A�* 

Average reward per stepz���T�       ��2	zl�̩�A�*

epsilonz���>�x�.       ��W�	fh�̩�A�* 

Average reward per stepz����
�n       ��2	Mi�̩�A�*

epsilonz����b�.       ��W�	�͚̩�A�* 

Average reward per stepz����{�       ��2	�Κ̩�A�*

epsilonz����.       ��W�	��̩�A�* 

Average reward per stepz�������       ��2	���̩�A�*

epsilonz���-FN^.       ��W�	���̩�A�* 

Average reward per stepz����V�T       ��2	���̩�A�*

epsilonz���h��.       ��W�	 �̩�A�* 

Average reward per stepz���{�n       ��2	��̩�A�*

epsilonz���|%�G.       ��W�	a4�̩�A�* 

Average reward per stepz����hsx       ��2	D5�̩�A�*

epsilonz���ZqjJ.       ��W�	���̩�A�* 

Average reward per stepz���q�       ��2	T��̩�A�*

epsilonz���J\M*.       ��W�	��̩�A�* 

Average reward per stepz�����3       ��2	���̩�A�*

epsilonz�����.       ��W�	@N�̩�A�* 

Average reward per stepz�������       ��2	"O�̩�A�*

epsilonz�����R.       ��W�	;q�̩�A�* 

Average reward per stepz���{��R       ��2	r�̩�A�*

epsilonz���d[�Z.       ��W�	�˭̩�A�* 

Average reward per stepz����v�       ��2	�̭̩�A�*

epsilonz���W��j.       ��W�	�ԯ̩�A�* 

Average reward per stepz���~�I~       ��2	�կ̩�A�*

epsilonz����Î@.       ��W�	� �̩�A�* 

Average reward per stepz�������       ��2	e�̩�A�*

epsilonz���U��%.       ��W�	�k�̩�A�* 

Average reward per stepz����!��       ��2	�l�̩�A�*

epsilonz�����˸0       ���_	���̩�A*#
!
Average reward per episode����霺�.       ��W�	T��̩�A*!

total reward per episode   ��+�.       ��W�	%y�̩�A�* 

Average reward per step����i�X�       ��2	�y�̩�A�*

epsilon�����O�9.       ��W�	�̩�A�* 

Average reward per step�������l       ��2	S�̩�A�*

epsilon�����K.       ��W�	��̩�A�* 

Average reward per step�����ߛp       ��2	2�̩�A�*

epsilon����
�U%.       ��W�	���̩�A�* 

Average reward per step�������       ��2	G��̩�A�*

epsilon����}�Gs.       ��W�	1x�̩�A�* 

Average reward per step����Y_�#       ��2	yy�̩�A�*

epsilon����0���.       ��W�	��̩�A�* 

Average reward per step�����o�q       ��2	���̩�A�*

epsilon�������.       ��W�	��̩�A�* 

Average reward per step�����z��       ��2	~�̩�A�*

epsilon����g �.       ��W�	"�̩�A�* 

Average reward per step������E~       ��2	�̩�A�*

epsilon����L �.       ��W�	�,�̩�A�* 

Average reward per step������       ��2	�-�̩�A�*

epsilon����Z�Ut.       ��W�	�7�̩�A�* 

Average reward per step������       ��2	�8�̩�A�*

epsilon����s�5.       ��W�	�K�̩�A�* 

Average reward per step�����"6?       ��2	�L�̩�A�*

epsilon����[33.       ��W�	gb�̩�A�* 

Average reward per step����O�Gr       ��2	�c�̩�A�*

epsilon����5cM.       ��W�		p�̩�A�* 

Average reward per step����ǲ2       ��2	�p�̩�A�*

epsilon����9��.       ��W�	�~�̩�A�* 

Average reward per step����Q~e�       ��2	,�̩�A�*

epsilon����(�r.       ��W�	T�̩�A�* 

Average reward per step�����x��       ��2	:�̩�A�*

epsilon������.       ��W�	5��̩�A�* 

Average reward per step�����<�       ��2	з�̩�A�*

epsilon�������p.       ��W�	Z��̩�A�* 

Average reward per step�����$��       ��2	���̩�A�*

epsilon�������.       ��W�	X��̩�A�* 

Average reward per step�����:n       ��2	��̩�A�*

epsilon�����w��.       ��W�	��̩�A�* 

Average reward per step����� B       ��2	���̩�A�*

epsilon����4��.       ��W�	\��̩�A�* 

Average reward per step����\�K�       ��2	���̩�A�*

epsilon�����F��.       ��W�	��̩�A�* 

Average reward per step����֟U�       ��2	��̩�A�*

epsilon��������.       ��W�	^��̩�A�* 

Average reward per step����Z9l       ��2	4��̩�A�*

epsilon����o��.       ��W�	7��̩�A�* 

Average reward per step�����\�       ��2	��̩�A�*

epsilon�����4��.       ��W�	���̩�A�* 

Average reward per step����� ��       ��2	���̩�A�*

epsilon�������C.       ��W�	���̩�A�* 

Average reward per step����#2       ��2	F��̩�A�*

epsilon����l,�.       ��W�	y]�̩�A�* 

Average reward per step����&c�       ��2	^�̩�A�*

epsilon�����o�f.       ��W�	��̩�A�* 

Average reward per step����_��G       ��2	{��̩�A�*

epsilon����ˣ>#.       ��W�	;��̩�A�* 

Average reward per step������(�       ��2	 �̩�A�*

epsilon����r�.       ��W�	1�̩�A�* 

Average reward per step����s�oY       ��2	+2�̩�A�*

epsilon�������.       ��W�	�l�̩�A�* 

Average reward per step����L�y       ��2	m�̩�A�*

epsilon����>RP�.       ��W�	���̩�A�* 

Average reward per step������9�       ��2	@��̩�A�*

epsilon���� �.       ��W�	b��̩�A�* 

Average reward per step����.W��       ��2	8��̩�A�*

epsilon����uq.0       ���_	���̩�A*#
!
Average reward per episode  ��vh��.       ��W�	q��̩�A*!

total reward per episode  �#y&H.       ��W�	�̩�A�* 

Average reward per step  ���VN�       ��2	ٔ̩�A�*

epsilon  �� ���.       ��W�	��̩�A�* 

Average reward per step  ���"�"       ��2	"�̩�A�*

epsilon  ����7.       ��W�	5~̩�A�* 

Average reward per step  ��[�a�       ��2	�~̩�A�*

epsilon  ��@�.       ��W�	��̩�A�* 

Average reward per step  ��z��       ��2	P�̩�A�*

epsilon  �����g.       ��W�	�	̩�A�* 

Average reward per step  ���-�       ��2	غ	̩�A�*

epsilon  �����\.       ��W�	��̩�A�* 

Average reward per step  ���@�       ��2	v�̩�A�*

epsilon  ��跑.       ��W�	�}̩�A�* 

Average reward per step  �����       ��2	�~̩�A�*

epsilon  ���V�&.       ��W�	b�̩�A�* 

Average reward per step  ���       ��2	8�̩�A�*

epsilon  ��-��.       ��W�	ƿ̩�A�* 

Average reward per step  ���٤       ��2	]�̩�A�*

epsilon  ����B@.       ��W�	��̩�A�* 

Average reward per step  ���4��       ��2	h�̩�A�*

epsilon  �����.       ��W�	n�̩�A�* 

Average reward per step  ��ղ�Q       ��2	�̩�A�*

epsilon  ��4��.       ��W�	��̩�A�* 

Average reward per step  ��+��       ��2	t�̩�A�*

epsilon  �����.       ��W�	x�̩�A�* 

Average reward per step  ��p�u�       ��2	c�̩�A�*

epsilon  �����.       ��W�	_
̩�A�* 

Average reward per step  ������       ��2	A̩�A�*

epsilon  ����h.       ��W�	Y2̩�A�* 

Average reward per step  ������       ��2	�2̩�A�*

epsilon  ��Xߕ�.       ��W�	�G ̩�A�* 

Average reward per step  ���-�       ��2	ZH ̩�A�*

epsilon  ���M*F.       ��W�	��!̩�A�* 

Average reward per step  ���l        ��2	��!̩�A�*

epsilon  ��r���.       ��W�	��$̩�A�* 

Average reward per step  ��)J�       ��2	�$̩�A�*

epsilon  ��;��].       ��W�	~U(̩�A�* 

Average reward per step  ��
�X       ��2	\V(̩�A�*

epsilon  ���Ǜ�.       ��W�	�*̩�A�* 

Average reward per step  ����T�       ��2	Ԃ*̩�A�*

epsilon  ���Dl7.       ��W�	�-̩�A�* 

Average reward per step  ���qa�       ��2	�-̩�A�*

epsilon  ���f6 .       ��W�	�t0̩�A�* 

Average reward per step  ���h
       ��2	yu0̩�A�*

epsilon  ��״5�0       ���_	��0̩�A*#
!
Average reward per episode�.���:^.       ��W�	!�0̩�A*!

total reward per episode  �͘��.       ��W�	֭4̩�A�* 

Average reward per step�.����       ��2	��4̩�A�*

epsilon�.��W:7.       ��W�	0�6̩�A�* 

Average reward per step�.��v�       ��2	��6̩�A�*

epsilon�.��>�A�.       ��W�	c�8̩�A�* 

Average reward per step�.������       ��2	��8̩�A�*

epsilon�.����	.       ��W�	��:̩�A�* 

Average reward per step�.��%���       ��2	��:̩�A�*

epsilon�.�����#.       ��W�	�=̩�A�* 

Average reward per step�.����]�       ��2	+=̩�A�*

epsilon�.���N(4.       ��W�	>̩�A�* 

Average reward per step�.���I       ��2	�>̩�A�*

epsilon�.��%X��.       ��W�	e�@̩�A�* 

Average reward per step�.�����       ��2	2�@̩�A�*

epsilon�.����jD.       ��W�	6�B̩�A�* 

Average reward per step�.���9�&       ��2	�B̩�A�*

epsilon�.���%�j.       ��W�	1�D̩�A�* 

Average reward per step�.���       ��2	1�D̩�A�*

epsilon�.��r��.       ��W�	��F̩�A�* 

Average reward per step�.��[S�[       ��2	��F̩�A�*

epsilon�.���4�.       ��W�	<�H̩�A�* 

Average reward per step�.��'-~�       ��2	Q�H̩�A�*

epsilon�.���U;D.       ��W�	K̩�A�* 

Average reward per step�.���R|       ��2	#K̩�A�*

epsilon�.����B�.       ��W�	�RM̩�A�* 

Average reward per step�.���ns       ��2	�SM̩�A�*

epsilon�.��l9�U.       ��W�	�N̩�A�* 

Average reward per step�.���I�       ��2	��N̩�A�*

epsilon�.���1��.       ��W�	��P̩�A�* 

Average reward per step�.�� �g�       ��2	��P̩�A�*

epsilon�.���T.E.       ��W�	��S̩�A�* 

Average reward per step�.����       ��2	ףS̩�A�*

epsilon�.��h{$E.       ��W�	�iU̩�A�* 

Average reward per step�.���9$       ��2	�jU̩�A�*

epsilon�.��3L�.       ��W�	�W̩�A�* 

Average reward per step�.���u\l       ��2	ͫW̩�A�*

epsilon�.����d�.       ��W�	�Y̩�A�* 

Average reward per step�.��:       ��2	��Y̩�A�*

epsilon�.��3|�.       ��W�	�![̩�A�* 

Average reward per step�.��6>[P       ��2	h"[̩�A�*

epsilon�.����!.       ��W�	�8]̩�A�* 

Average reward per step�.��&�\�       ��2	�9]̩�A�*

epsilon�.���BR\.       ��W�	��`̩�A�* 

Average reward per step�.�����`       ��2	��`̩�A�*

epsilon�.��%G.       ��W�	�Fc̩�A�* 

Average reward per step�.���?�Y       ��2	kGc̩�A�*

epsilon�.���&�20       ���_	�fc̩�A *#
!
Average reward per episode-d����n�.       ��W�	{gc̩�A *!

total reward per episode  "�a��.       ��W�	��g̩�A�* 

Average reward per step-d���o��       ��2	7�g̩�A�*

epsilon-d��n��.       ��W�	��i̩�A�* 

Average reward per step-d��-Pf       ��2	��i̩�A�*

epsilon-d��
Ȭl.       ��W�	-l̩�A�* 

Average reward per step-d�� �[�       ��2	�l̩�A�*

epsilon-d��轾Q.       ��W�	n5n̩�A�* 

Average reward per step-d�����z       ��2	?6n̩�A�*

epsilon-d��$GUQ.       ��W�	�]p̩�A�* 

Average reward per step-d���{SD       ��2	c^p̩�A�*

epsilon-d����F>.       ��W�	}�q̩�A�* 

Average reward per step-d��B�:0       ��2	K�q̩�A�*

epsilon-d��Τ u.       ��W�	L7t̩�A�* 

Average reward per step-d���g�       ��2	28t̩�A�*

epsilon-d�����I.       ��W�	M�u̩�A�* 

Average reward per step-d���mg�       ��2	H�u̩�A�*

epsilon-d�����.       ��W�	��w̩�A�* 

Average reward per step-d����Z�       ��2	t�w̩�A�*

epsilon-d�����.       ��W�	z̩�A�* 

Average reward per step-d�����       ��2		z̩�A�*

epsilon-d����.       ��W�	�*|̩�A�* 

Average reward per step-d�����       ��2	�+|̩�A�*

epsilon-d���4V.       ��W�	�c~̩�A�* 

Average reward per step-d���s7       ��2	^d~̩�A�*

epsilon-d������.       ��W�	���̩�A�* 

Average reward per step-d�����       ��2	���̩�A�*

epsilon-d��1ز�.       ��W�	�Q�̩�A�* 

Average reward per step-d����2       ��2	�R�̩�A�*

epsilon-d���}/�0       ���_	n��̩�A!*#
!
Average reward per episoden�>�m�K�.       ��W�	;��̩�A!*!

total reward per episode  '����*.       ��W�	^��̩�A�* 

Average reward per stepn�>�C�	�       ��2	8��̩�A�*

epsilonn�>�w`��.       ��W�	�A�̩�A�* 

Average reward per stepn�>�Q� 6       ��2	�B�̩�A�*

epsilonn�>��&=H.       ��W�	"�̩�A�* 

Average reward per stepn�>��
_�       ��2	�"�̩�A�*

epsilonn�>���^�.       ��W�	2��̩�A�* 

Average reward per stepn�>��P       ��2	��̩�A�*

epsilonn�>���q.       ��W�	�ؐ̩�A�* 

Average reward per stepn�>�����       ��2	�ِ̩�A�*

epsilonn�>�Q��.       ��W�	,��̩�A�* 

Average reward per stepn�>�g� �       ��2	���̩�A�*

epsilonn�>�]+A�.       ��W�	�B�̩�A�* 

Average reward per stepn�>���*       ��2	�C�̩�A�*

epsilonn�>��p�.       ��W�	QM�̩�A�* 

Average reward per stepn�>��~(
       ��2	3N�̩�A�*

epsilonn�>�^�\.       ��W�	��̩�A�* 

Average reward per stepn�>�Mp�       ��2	[�̩�A�*

epsilonn�>�e�Ѭ.       ��W�	�E�̩�A�* 

Average reward per stepn�>�k�%       ��2	�F�̩�A�*

epsilonn�>�8Jw`.       ��W�	�o�̩�A�* 

Average reward per stepn�>����       ��2	ap�̩�A�*

epsilonn�>�u?�L.       ��W�	���̩�A�* 

Average reward per stepn�>��[̩       ��2	���̩�A�*

epsilonn�>���k�.       ��W�	��̩�A�* 

Average reward per stepn�>���k�       ��2	��̩�A�*

epsilonn�>�7�.       ��W�	�%�̩�A�* 

Average reward per stepn�>���X�       ��2	�&�̩�A�*

epsilonn�>�9Yo.       ��W�	\�̩�A�* 

Average reward per stepn�>���.       ��2	�\�̩�A�*

epsilonn�>���͎.       ��W�	ԁ�̩�A�* 

Average reward per stepn�>����[       ��2	���̩�A�*

epsilonn�>��Z
L.       ��W�	a��̩�A�* 

Average reward per stepn�>��%�       ��2	��̩�A�*

epsilonn�>�PC��.       ��W�	i��̩�A�* 

Average reward per stepn�>�$��       ��2	S��̩�A�*

epsilonn�>��h�.       ��W�	���̩�A�* 

Average reward per stepn�>�����       ��2	E��̩�A�*

epsilonn�>���i-.       ��W�	�'�̩�A�* 

Average reward per stepn�>����       ��2	(�̩�A�*

epsilonn�>�p��$.       ��W�	�I�̩�A�* 

Average reward per stepn�>�!�h�       ��2	MJ�̩�A�*

epsilonn�>�1���.       ��W�	1{�̩�A�* 

Average reward per stepn�>�e�De       ��2	|�̩�A�*

epsilonn�>�ҋ��.       ��W�	Tt�̩�A�* 

Average reward per stepn�>��;�       ��2	u�̩�A�*

epsilonn�>��؏�.       ��W�	"5�̩�A�* 

Average reward per stepn�>��x�{       ��2	�5�̩�A�*

epsilonn�>�#�s.       ��W�	Л�̩�A�* 

Average reward per stepn�>�D�YI       ��2	x��̩�A�*

epsilonn�>�A��&.       ��W�	w��̩�A�* 

Average reward per stepn�>�t�       ��2	Y��̩�A�*

epsilonn�>���.       ��W�	)�̩�A�* 

Average reward per stepn�>��V:       ��2	:�̩�A�*

epsilonn�>����0       ���_	L�̩�A"*#
!
Average reward per episode������.       ��W�	��̩�A"*!

total reward per episode  �M%=l.       ��W�	���̩�A�* 

Average reward per step����G�#       ��2	b��̩�A�*

epsilon����&!Ut.       ��W�	~��̩�A�* 

Average reward per step����w�~f       ��2	P��̩�A�*

epsilon����/|s.       ��W�	��̩�A�* 

Average reward per step�����d       ��2	��̩�A�*

epsilon����fI.       ��W�	�
�̩�A�* 

Average reward per step�����un�       ��2	p�̩�A�*

epsilon����� .       ��W�	��̩�A�* 

Average reward per step������v�       ��2	m �̩�A�*

epsilon����D�W�.       ��W�	�-�̩�A�* 

Average reward per step����SR;       ��2	I.�̩�A�*

epsilon����k�.       ��W�	#M�̩�A�* 

Average reward per step����A8�O       ��2	�M�̩�A�*

epsilon�������c.       ��W�	���̩�A�* 

Average reward per step�����}	�       ��2	q��̩�A�*

epsilon����I,��.       ��W�	���̩�A�* 

Average reward per step�����:�R       ��2	���̩�A�*

epsilon����k�f.       ��W�	![�̩�A�* 

Average reward per step������       ��2	�[�̩�A�*

epsilon����왍V.       ��W�	��̩�A�* 

Average reward per step����m��o       ��2	��̩�A�*

epsilon�����2�.       ��W�	�E�̩�A�* 

Average reward per step����Ƴ       ��2	xF�̩�A�*

epsilon����z�o}.       ��W�	��̩�A�* 

Average reward per step����h�N       ��2	��̩�A�*

epsilon�����@e.       ��W�	�'�̩�A�* 

Average reward per step����u��~       ��2	k(�̩�A�*

epsilon�������.       ��W�	�-�̩�A�* 

Average reward per step������,�       ��2	o.�̩�A�*

epsilon����DD��.       ��W�	U�̩�A�* 

Average reward per step����.�a�       ��2	�U�̩�A�*

epsilon�����|��.       ��W�	��̩�A�* 

Average reward per step��������       ��2	��̩�A�*

epsilon����-|�=.       ��W�	�+�̩�A�* 

Average reward per step������       ��2	o,�̩�A�*

epsilon�����\��.       ��W�	�+�̩�A�* 

Average reward per step�����7��       ��2	�,�̩�A�*

epsilon������7�.       ��W�	�d�̩�A�* 

Average reward per step����E�       ��2	�e�̩�A�*

epsilon����G��.       ��W�	���̩�A�* 

Average reward per step����/8�       ��2	���̩�A�*

epsilon������+.       ��W�	��̩�A�* 

Average reward per step�����Ʒ�       ��2	ܼ�̩�A�*

epsilon�����!^-.       ��W�	$�̩�A�* 

Average reward per step�������       ��2	�$�̩�A�*

epsilon����$�@�.       ��W�	VD�̩�A�* 

Average reward per step�����f:       ��2	1E�̩�A�*

epsilon����.ᇳ.       ��W�	Զ�̩�A�* 

Average reward per step�����H�       ��2	���̩�A�*

epsilon������A.       ��W�	�.�̩�A�* 

Average reward per step�����R��       ��2	�/�̩�A�*

epsilon����b��c.       ��W�	�a�̩�A�* 

Average reward per step�����@��       ��2	�b�̩�A�*

epsilon����1�8�.       ��W�	��̩�A�* 

Average reward per step�����	�       ��2	���̩�A�*

epsilon����	Y�.       ��W�	+��̩�A�* 

Average reward per step����J���       ��2	���̩�A�*

epsilon�����<��.       ��W�	���̩�A�* 

Average reward per step�������b       ��2	���̩�A�*

epsilon����{3Ӫ.       ��W�	�m�̩�A�* 

Average reward per step�����P�-       ��2	�n�̩�A�*

epsilon�����¿h.       ��W�	P�̩�A�* 

Average reward per step����?�X,       ��2	�̩�A�*

epsilon������!.       ��W�	��̩�A�* 

Average reward per step����F�|S       ��2	��̩�A�*

epsilon�����p�.       ��W�	$�̩�A�* 

Average reward per step����-�2�       ��2	��̩�A�*

epsilon�����^.       ��W�	%̩�A�* 

Average reward per step�����_�       ��2	�̩�A�*

epsilon����w��z.       ��W�	
̩�A�* 

Average reward per step�������H       ��2	�
̩�A�*

epsilon����Yk��.       ��W�	��̩�A�* 

Average reward per step������       ��2	̷̩�A�*

epsilon�����WVv.       ��W�	��̩�A�* 

Average reward per step����J)c       ��2	��̩�A�*

epsilon����I}d.       ��W�	Z)̩�A�* 

Average reward per step������˜       ��2	0*̩�A�*

epsilon����!�j�.       ��W�	 �̩�A�* 

Average reward per step����´a�       ��2	��̩�A�*

epsilon�����K.       ��W�	�K̩�A�* 

Average reward per step����i��(       ��2	�L̩�A�*

epsilon�������a.       ��W�	��̩�A�* 

Average reward per step����d�       ��2	U�̩�A�*

epsilon�����ʉ.       ��W�	�̩�A�* 

Average reward per step�����.
�       ��2	�̩�A�*

epsilon�������.       ��W�	4�̩�A�* 

Average reward per step����j�q�       ��2	�̩�A�*

epsilon����ĳ�.       ��W�	̩�A�* 

Average reward per step�����       ��2	�̩�A�*

epsilon�����ǡ.       ��W�	�(̩�A�* 

Average reward per step��������       ��2	�)̩�A�*

epsilon������2�.       ��W�	�a ̩�A�* 

Average reward per step�����uX�       ��2	�b ̩�A�*

epsilon����0,[�.       ��W�	�!̩�A�* 

Average reward per step������       ��2	7�!̩�A�*

epsilon����:��W.       ��W�	�T#̩�A�* 

Average reward per step�����[�9       ��2	�U#̩�A�*

epsilon����.IB�.       ��W�	�c%̩�A�* 

Average reward per step�����,�       ��2	|d%̩�A�*

epsilon����P�.       ��W�	Ɗ'̩�A�* 

Average reward per step����t;Y�       ��2	��'̩�A�*

epsilon�����X.       ��W�	��)̩�A�* 

Average reward per step������I/       ��2	��)̩�A�*

epsilon����v�i�.       ��W�	��+̩�A�* 

Average reward per step�����/��       ��2	��+̩�A�*

epsilon����7���.       ��W�	�-̩�A�* 

Average reward per step������)�       ��2	��-̩�A�*

epsilon����*ɶ�.       ��W�	��/̩�A�* 

Average reward per step������M       ��2	��/̩�A�*

epsilon�����V�'.       ��W�	�x1̩�A�* 

Average reward per step�����p4�       ��2	�y1̩�A�*

epsilon������.       ��W�	�3̩�A�* 

Average reward per step����b�6       ��2	��3̩�A�*

epsilon������]j.       ��W�	��5̩�A�* 

Average reward per step������J$       ��2	X�5̩�A�*

epsilon������l.       ��W�	8̩�A�* 

Average reward per step����K��       ��2	�8̩�A�*

epsilon�����k �.       ��W�	&�9̩�A�* 

Average reward per step�����Z	       ��2	?�9̩�A�*

epsilon�����d�.       ��W�	�2<̩�A�* 

Average reward per step�������Y       ��2	�3<̩�A�*

epsilon����Z_V.       ��W�	��=̩�A�* 

Average reward per step�����6�       ��2	V�=̩�A�*

epsilon����F��.       ��W�	��?̩�A�* 

Average reward per step����u�(       ��2	��?̩�A�*

epsilon������yy.       ��W�	AB̩�A�* 

Average reward per step�����MH�       ��2	9BB̩�A�*

epsilon�����#�0       ���_	�]B̩�A#*#
!
Average reward per episode  d��.       ��W�	N^B̩�A#*!

total reward per episode  d�i�_4.       ��W�	Z�E̩�A�* 

Average reward per step  d�@2��       ��2	��E̩�A�*

epsilon  d�@^%.       ��W�	4H̩�A�* 

Average reward per step  d��e-%       ��2	�4H̩�A�*

epsilon  d��/ig.       ��W�	��I̩�A�* 

Average reward per step  d���'�       ��2	��I̩�A�*

epsilon  d�e[K=.       ��W�	��L̩�A�* 

Average reward per step  d�aЌF       ��2	e�L̩�A�*

epsilon  d����.       ��W�	*�N̩�A�* 

Average reward per step  d����       ��2	��N̩�A�*

epsilon  d���.       ��W�	D1Q̩�A�* 

Average reward per step  d���#       ��2	�1Q̩�A�*

epsilon  d��}�`.       ��W�	��R̩�A�* 

Average reward per step  d��#*�       ��2	J�R̩�A�*

epsilon  d����i.       ��W�	j�T̩�A�* 

Average reward per step  d�a>~�       ��2	�T̩�A�*

epsilon  d�i���.       ��W�	$�W̩�A�* 

Average reward per step  d�H-_�       ��2	�W̩�A�*

epsilon  d��}�U.       ��W�	�yZ̩�A�* 

Average reward per step  d��h�       ��2	�zZ̩�A�*

epsilon  d��CD�.       ��W�	Z�[̩�A�* 

Average reward per step  d�W�!       ��2	#�[̩�A�*

epsilon  d�U+6�.       ��W�	�6^̩�A�* 

Average reward per step  d��p�       ��2	~7^̩�A�*

epsilon  d�w�D.       ��W�	֧_̩�A�* 

Average reward per step  d�� ל       ��2	��_̩�A�*

epsilon  d�g	��.       ��W�	�]b̩�A�* 

Average reward per step  d�q�       ��2	�^b̩�A�*

epsilon  d�Ҁ�|.       ��W�	�c̩�A�* 

Average reward per step  d�����       ��2	��c̩�A�*

epsilon  d�e�.       ��W�	ݕf̩�A�* 

Average reward per step  d����       ��2	��f̩�A�*

epsilon  d��Y.       ��W�	��h̩�A�* 

Average reward per step  d���<       ��2	��h̩�A�*

epsilon  d����.       ��W�	ٙl̩�A�* 

Average reward per step  d�Î��       ��2	$�l̩�A�*

epsilon  d�ᶙO0       ���_	�l̩�A$*#
!
Average reward per episode�8�c�Wc.       ��W�	�l̩�A$*!

total reward per episode  ��7x .       ��W�	'�p̩�A�* 

Average reward per step�8�n�;q       ��2	+�p̩�A�*

epsilon�8�����.       ��W�	��r̩�A�* 

Average reward per step�8����       ��2	u�r̩�A�*

epsilon�8��g�P.       ��W�	�/t̩�A�* 

Average reward per step�8��O>       ��2	^0t̩�A�*

epsilon�8�Qq�.       ��W�	w̩�A�* 

Average reward per step�8��}�`       ��2	�w̩�A�*

epsilon�8���c/.       ��W�	8�x̩�A�* 

Average reward per step�8��qǘ       ��2	��x̩�A�*

epsilon�8���.       ��W�	��z̩�A�* 

Average reward per step�8�d��g       ��2	t�z̩�A�*

epsilon�8��F){.       ��W�	?�|̩�A�* 

Average reward per step�8����W       ��2	ߌ|̩�A�*

epsilon�8���.       ��W�	:�~̩�A�* 

Average reward per step�8�E�R       ��2	��~̩�A�*

epsilon�8����.       ��W�	:W�̩�A�* 

Average reward per step�8�|��U       ��2	X�̩�A�*

epsilon�8�?d�.       ��W�	�R�̩�A�* 

Average reward per step�8�_�H�       ��2	�S�̩�A�*

epsilon�8�Zvu8.       ��W�	⬄̩�A�	* 

Average reward per step�8����s       ��2	���̩�A�	*

epsilon�8�bΡU.       ��W�	�ņ̩�A�	* 

Average reward per step�8��H�V       ��2	�Ɔ̩�A�	*

epsilon�8�t��.       ��W�	dv�̩�A�	* 

Average reward per step�8���       ��2	�v�̩�A�	*

epsilon�8�%��	.       ��W�	̊̩�A�	* 

Average reward per step�8���       ��2	�̊̩�A�	*

epsilon�8�.�(�.       ��W�	u�̩�A�	* 

Average reward per step�8�����       ��2	?�̩�A�	*

epsilon�8�5�3�.       ��W�	��̩�A�	* 

Average reward per step�8��G�	       ��2	�̩�A�	*

epsilon�8�����.       ��W�	t%�̩�A�	* 

Average reward per step�8��.��       ��2	&�̩�A�	*

epsilon�8���9V.       ��W�	0�̩�A�	* 

Average reward per step�8�(�Z       ��2	��̩�A�	*

epsilon�8�~�#.       ��W�	<�̩�A�	* 

Average reward per step�8��(K       ��2	
�̩�A�	*

epsilon�8��y`e.       ��W�	Z)�̩�A�	* 

Average reward per step�8��fN�       ��2	=*�̩�A�	*

epsilon�8��:�.       ��W�	8L�̩�A�	* 

Average reward per step�8�Y�W�       ��2	+M�̩�A�	*

epsilon�8�>��.       ��W�	�Y�̩�A�	* 

Average reward per step�8�z�Iv       ��2	>Z�̩�A�	*

epsilon�8���w�.       ��W�	�q�̩�A�	* 

Average reward per step�8���CJ       ��2	\r�̩�A�	*

epsilon�8��%d0       ���_	���̩�A%*#
!
Average reward per episode�����7a.       ��W�	���̩�A%*!

total reward per episode  ��J��.       ��W�	p{�̩�A�	* 

Average reward per step����`ˉq       ��2	c|�̩�A�	*

epsilon������N�.       ��W�	��̩�A�	* 

Average reward per step����0�A       ��2	߈�̩�A�	*

epsilon����U�͏.       ��W�	�ؤ̩�A�	* 

Average reward per step��������       ��2	�٤̩�A�	*

epsilon����:�V\.       ��W�	��̩�A�	* 

Average reward per step������v       ��2	��̩�A�	*

epsilon����њQ.       ��W�	���̩�A�	* 

Average reward per step������Z       ��2	{��̩�A�	*

epsilon�����&�R.       ��W�	�֬̩�A�	* 

Average reward per step����rb       ��2	4׬̩�A�	*

epsilon����7�s�.       ��W�	�[�̩�A�	* 

Average reward per step����jvkT       ��2	�\�̩�A�	*

epsilon����Һ��.       ��W�	c�̩�A�	* 

Average reward per step�����:       ��2	E�̩�A�	*

epsilon����w�ˏ.       ��W�	mt�̩�A�	* 

Average reward per step����{�m�       ��2	%u�̩�A�	*

epsilon�����s�.       ��W�	;�̩�A�	* 

Average reward per step������X6       ��2	�̩�A�	*

epsilon����^mk3.       ��W�	��̩�A�	* 

Average reward per step�����pK�       ��2	g�̩�A�	*

epsilon������΂.       ��W�	��̩�A�	* 

Average reward per step����3|t       ��2	�	�̩�A�	*

epsilon����h��.       ��W�	5��̩�A�	* 

Average reward per step����6/c       ��2	��̩�A�	*

epsilon����&K�+.       ��W�	r�̩�A�	* 

Average reward per step������;.       ��2	L�̩�A�	*

epsilon������<d.       ��W�	�4�̩�A�	* 

Average reward per step����I�       ��2	r5�̩�A�	*

epsilon����~1�.       ��W�	U�̩�A�	* 

Average reward per step�����;�6       ��2	�U�̩�A�	*

epsilon������`�.       ��W�	n��̩�A�	* 

Average reward per step����(K;j       ��2	D��̩�A�	*

epsilon������(].       ��W�	���̩�A�	* 

Average reward per step����ȫY       ��2	���̩�A�	*

epsilon����p��.       ��W�	g�̩�A�	* 

Average reward per step����ݴA       ��2	Zh�̩�A�	*

epsilon����S�-.       ��W�	�1�̩�A�	* 

Average reward per step�����r�       ��2	U2�̩�A�	*

epsilon���� �/�.       ��W�	��̩�A�	* 

Average reward per step����(=��       ��2	��̩�A�	*

epsilon������+-.       ��W�	\��̩�A�	* 

Average reward per step�������       ��2	���̩�A�	*

epsilon������|�.       ��W�	���̩�A�	* 

Average reward per step�����*v�       ��2	)��̩�A�	*

epsilon����H�/�.       ��W�	�y�̩�A�	* 

Average reward per step�������:       ��2	-{�̩�A�	*

epsilon�����S�`.       ��W�	f1�̩�A�	* 

Average reward per step����΅�       ��2	D2�̩�A�	*

epsilon����v**.       ��W�	��̩�A�	* 

Average reward per step����b�l       ��2	���̩�A�	*

epsilon�����!�.       ��W�	]��̩�A�	* 

Average reward per step������[�       ��2	#��̩�A�	*

epsilon����Y���.       ��W�	`#�̩�A�	* 

Average reward per step������ن       ��2	:$�̩�A�	*

epsilon������z.       ��W�	�#�̩�A�	* 

Average reward per step����Q�1�       ��2	)$�̩�A�	*

epsilon�����4*z.       ��W�	0F�̩�A�	* 

Average reward per step������ �       ��2	sG�̩�A�	*

epsilon�������.       ��W�	r��̩�A�	* 

Average reward per step����u�L�       ��2	D��̩�A�	*

epsilon�����G�.       ��W�	�̩�A�	* 

Average reward per step������iV       ��2	��̩�A�	*

epsilon������.       ��W�	W��̩�A�	* 

Average reward per step����14b       ��2	9��̩�A�	*

epsilon�����6��.       ��W�	�f�̩�A�	* 

Average reward per step����E�0       ��2	0g�̩�A�	*

epsilon�����3b0       ���_	:��̩�A&*#
!
Average reward per episodeiiI��S�G.       ��W�	��̩�A&*!

total reward per episode  ��^�e�.       ��W�	�3�̩�A�	* 

Average reward per stepiiI�\�f       ��2	�4�̩�A�	*

epsiloniiI�4��.       ��W�	4��̩�A�	* 

Average reward per stepiiI�9L       ��2	���̩�A�	*

epsiloniiI����.       ��W�	6=�̩�A�	* 

Average reward per stepiiI�(���       ��2	>�̩�A�	*

epsiloniiI�P9
�.       ��W�	�G�̩�A�	* 

Average reward per stepiiI�|�       ��2	^H�̩�A�	*

epsiloniiI����f.       ��W�	�l�̩�A�	* 

Average reward per stepiiI���B�       ��2	�m�̩�A�	*

epsiloniiI�ŒK�.       ��W�	���̩�A�	* 

Average reward per stepiiI�7H�       ��2	���̩�A�	*

epsiloniiI����.       ��W�	���̩�A�	* 

Average reward per stepiiI��G��       ��2	ú�̩�A�	*

epsiloniiI��g��.       ��W�	���̩�A�	* 

Average reward per stepiiI�L�w&       ��2	s��̩�A�	*

epsiloniiI��c.       ��W�	R|�̩�A�	* 

Average reward per stepiiI���%�       ��2	�|�̩�A�	*

epsiloniiI�'���.       ��W�	��̩�A�	* 

Average reward per stepiiI�iүM       ��2	s�̩�A�	*

epsiloniiI�Zl6.       ��W�	n�̩�A�	* 

Average reward per stepiiI����`       ��2	+�̩�A�	*

epsiloniiI�7��A.       ��W�	�$̩�A�	* 

Average reward per stepiiI��$�5       ��2	�%̩�A�	*

epsiloniiI��ws�.       ��W�	z7̩�A�	* 

Average reward per stepiiI�Y[Q�       ��2	�8̩�A�	*

epsiloniiI��O��.       ��W�	��	̩�A�	* 

Average reward per stepiiI�YPi       ��2	X�	̩�A�	*

epsiloniiI����V.       ��W�	%�̩�A�	* 

Average reward per stepiiI�2Dܰ       ��2	!�̩�A�	*

epsiloniiI�p�Np.       ��W�	P̩�A�	* 

Average reward per stepiiI�rU!/       ��2	"̩�A�	*

epsiloniiI����.       ��W�	�a̩�A�	* 

Average reward per stepiiI���_       ��2	�b̩�A�	*

epsiloniiI���A/.       ��W�	��̩�A�	* 

Average reward per stepiiI�z6}       ��2	q�̩�A�	*

epsiloniiI�`�n0.       ��W�	Qi̩�A�	* 

Average reward per stepiiI��ҩ       ��2	Dj̩�A�	*

epsiloniiI�>�-.       ��W�	��̩�A�	* 

Average reward per stepiiI��u�t       ��2	��̩�A�	*

epsiloniiI�0�J�.       ��W�	~�̩�A�	* 

Average reward per stepiiI�(��v       ��2	 ̩�A�	*

epsiloniiI��O�.       ��W�	J̩�A�	* 

Average reward per stepiiI�&4t�       ��2	̩�A�	*

epsiloniiI��v.       ��W�	*̩�A�	* 

Average reward per stepiiI�n�U[       ��2	�̩�A�	*

epsiloniiI��,Cj.       ��W�	.�̩�A�	* 

Average reward per stepiiI��6�y       ��2	��̩�A�	*

epsiloniiI�5j��.       ��W�	m�̩�A�	* 

Average reward per stepiiI�4<v�       ��2	%�̩�A�	*

epsiloniiI���k.       ��W�	�� ̩�A�	* 

Average reward per stepiiI�t��g       ��2	r� ̩�A�	*

epsiloniiI����.       ��W�	}"̩�A�	* 

Average reward per stepiiI�]�M       ��2	�}"̩�A�	*

epsiloniiI� ��.       ��W�	��$̩�A�	* 

Average reward per stepiiI�U�ɉ       ��2	c�$̩�A�	*

epsiloniiI�	��.       ��W�	)'̩�A�	* 

Average reward per stepiiI��[�z       ��2	'̩�A�	*

epsiloniiI�~�r�.       ��W�	#)̩�A�	* 

Average reward per stepiiI�+!��       ��2	�)̩�A�	*

epsiloniiI���t.       ��W�	�+̩�A�	* 

Average reward per stepiiI��ol�       ��2	�+̩�A�	*

epsiloniiI��޾�.       ��W�	�)-̩�A�	* 

Average reward per stepiiI�rl�       ��2	k*-̩�A�	*

epsiloniiI�dL(�.       ��W�		N/̩�A�	* 

Average reward per stepiiI�@�>�       ��2	�N/̩�A�	*

epsiloniiI��Z��.       ��W�	3m1̩�A�	* 

Average reward per stepiiI��,       ��2	Dn1̩�A�	*

epsiloniiI��.       ��W�	f�2̩�A�	* 

Average reward per stepiiI�z"l       ��2	I�2̩�A�	*

epsiloniiI��X�.       ��W�	�4̩�A�	* 

Average reward per stepiiI�ƕ�       ��2	��4̩�A�	*

epsiloniiI�8�ߪ.       ��W�	��6̩�A�	* 

Average reward per stepiiI����j       ��2	��6̩�A�	*

epsiloniiI�<}:�.       ��W�	�_8̩�A�	* 

Average reward per stepiiI�+)�;       ��2	W`8̩�A�	*

epsiloniiI��W9!.       ��W�	��:̩�A�	* 

Average reward per stepiiI�Z wR       ��2	�:̩�A�	*

epsiloniiI�%��.       ��W�	��<̩�A�	* 

Average reward per stepiiI��o(L       ��2	R�<̩�A�	*

epsiloniiI���.       ��W�	�?̩�A�	* 

Average reward per stepiiI�	��       ��2	�?̩�A�	*

epsiloniiI�@6CZ.       ��W�	#NA̩�A�	* 

Average reward per stepiiI��ͦ�       ��2	�NA̩�A�	*

epsiloniiI�Uq�J.       ��W�	��D̩�A�	* 

Average reward per stepiiI����       ��2	�D̩�A�	*

epsiloniiI���f.       ��W�	s�F̩�A�	* 

Average reward per stepiiI����       ��2	Q�F̩�A�	*

epsiloniiI��v	<.       ��W�	�H̩�A�	* 

Average reward per stepiiI�S�)�       ��2	�H̩�A�	*

epsiloniiI�T
�..       ��W�	��J̩�A�	* 

Average reward per stepiiI��	��       ��2	��J̩�A�	*

epsiloniiI��p�b.       ��W�	�M̩�A�	* 

Average reward per stepiiI�H'{~       ��2	�M̩�A�	*

epsiloniiI����.       ��W�	�O̩�A�	* 

Average reward per stepiiI�>��7       ��2	�O̩�A�	*

epsiloniiI��C˹.       ��W�	�P̩�A�	* 

Average reward per stepiiI�"���       ��2	��P̩�A�	*

epsiloniiI�i�
y.       ��W�	�R̩�A�	* 

Average reward per stepiiI�X
�       ��2	�R̩�A�	*

epsiloniiI��N�H.       ��W�	�7T̩�A�	* 

Average reward per stepiiI���y       ��2	a8T̩�A�	*

epsiloniiI����.       ��W�	)uV̩�A�	* 

Average reward per stepiiI�c��       ��2	�uV̩�A�	*

epsiloniiI�:dN[.       ��W�	P�W̩�A�	* 

Average reward per stepiiI��w�       ��2	��W̩�A�	*

epsiloniiI���P.       ��W�	��Z̩�A�	* 

Average reward per stepiiI��`�w       ��2	z�Z̩�A�	*

epsiloniiI���Q8.       ��W�	� \̩�A�	* 

Average reward per stepiiI�ۃ�       ��2	>!\̩�A�	*

epsiloniiI���� .       ��W�	�t^̩�A�	* 

Average reward per stepiiI�D��P       ��2	`u^̩�A�	*

epsiloniiI�0�p.       ��W�	��`̩�A�	* 

Average reward per stepiiI�C�@�       ��2	T�`̩�A�	*

epsiloniiI��F�.       ��W�	��b̩�A�	* 

Average reward per stepiiI�um��       ��2	]�b̩�A�	*

epsiloniiI��`i�.       ��W�	��d̩�A�	* 

Average reward per stepiiI�O:3�       ��2	��d̩�A�	*

epsiloniiI��Xf.       ��W�	�g̩�A�	* 

Average reward per stepiiI�P�       ��2	�g̩�A�	*

epsiloniiI�����.       ��W�	*sh̩�A�	* 

Average reward per stepiiI��Cϲ       ��2	th̩�A�	*

epsiloniiI�Eo�.       ��W�	�j̩�A�	* 

Average reward per stepiiI����       ��2	ƿj̩�A�	*

epsiloniiI��L=f.       ��W�	l�l̩�A�	* 

Average reward per stepiiI��ّS       ��2	[�l̩�A�	*

epsiloniiI��t!�.       ��W�	$&o̩�A�	* 

Average reward per stepiiI��-�r       ��2	�&o̩�A�	*

epsiloniiI��$��.       ��W�	X�p̩�A�	* 

Average reward per stepiiI��G�Z       ��2	C�p̩�A�	*

epsiloniiI�I�.       ��W�	�s̩�A�	* 

Average reward per stepiiI�2       ��2	�s̩�A�	*

epsiloniiI�0nV.       ��W�	�.u̩�A�	* 

Average reward per stepiiI��E]�       ��2	�/u̩�A�	*

epsiloniiI���%�.       ��W�	B?w̩�A�	* 

Average reward per stepiiI�v�WT       ��2	�?w̩�A�	*

epsiloniiI�����.       ��W�	�x̩�A�	* 

Average reward per stepiiI�n�[       ��2	��x̩�A�	*

epsiloniiI�8�u�.       ��W�	��z̩�A�	* 

Average reward per stepiiI�ᜓP       ��2	��z̩�A�	*

epsiloniiI�<Vx�.       ��W�	��|̩�A�	* 

Average reward per stepiiI��m�
       ��2	��|̩�A�	*

epsiloniiI�U���.       ��W�	�Q̩�A�	* 

Average reward per stepiiI�|��       ��2	aR̩�A�	*

epsiloniiI����^.       ��W�	�׀̩�A�	* 

Average reward per stepiiI�a��N       ��2	E؀̩�A�	*

epsiloniiI����.       ��W�	b��̩�A�	* 

Average reward per stepiiI��a�       ��2	Q��̩�A�	*

epsiloniiI���}.       ��W�	��̩�A�	* 

Average reward per stepiiI��(i�       ��2	� �̩�A�	*

epsiloniiI�l?U�.       ��W�	NC�̩�A�	* 

Average reward per stepiiI��/�       ��2	 D�̩�A�	*

epsiloniiI��I.       ��W�	�m�̩�A�	* 

Average reward per stepiiI����-       ��2	�n�̩�A�	*

epsiloniiI�:��.       ��W�	Q��̩�A�	* 

Average reward per stepiiI�����       ��2	r��̩�A�	*

epsiloniiI��%d.       ��W�	��̩�A�	* 

Average reward per stepiiI�~h��       ��2	��̩�A�	*

epsiloniiI�9B�.       ��W�	W��̩�A�	* 

Average reward per stepiiI��;�       ��2	(��̩�A�	*

epsiloniiI���.       ��W�	��̩�A�	* 

Average reward per stepiiI���P       ��2	��̩�A�	*

epsiloniiI���?�.       ��W�	�E�̩�A�
* 

Average reward per stepiiI�>�       ��2	kF�̩�A�
*

epsiloniiI��6ӣ.       ��W�	���̩�A�
* 

Average reward per stepiiI����       ��2	i��̩�A�
*

epsiloniiI�y�_�.       ��W�	H�̩�A�
* 

Average reward per stepiiI�_T&4       ��2	8I�̩�A�
*

epsiloniiI���1[.       ��W�	�ɝ̩�A�
* 

Average reward per stepiiI��@"*       ��2	�ʝ̩�A�
*

epsiloniiI�R��o.       ��W�	0�̩�A�
* 

Average reward per stepiiI���V�       ��2	��̩�A�
*

epsiloniiI�׾�.       ��W�	��̩�A�
* 

Average reward per stepiiI�Ŏ(�       ��2	���̩�A�
*

epsiloniiI�x�5.       ��W�	�Σ̩�A�
* 

Average reward per stepiiI�
���       ��2	У̩�A�
*

epsiloniiI���L.       ��W�	'��̩�A�
* 

Average reward per stepiiI�i뉮       ��2	���̩�A�
*

epsiloniiI�q��F.       ��W�	��̩�A�
* 

Average reward per stepiiI�SZC�       ��2	��̩�A�
*

epsiloniiI��#R�.       ��W�	�L�̩�A�
* 

Average reward per stepiiI��4�
       ��2	IM�̩�A�
*

epsiloniiI��$[R.       ��W�	�ի̩�A�
* 

Average reward per stepiiI�U}�       ��2	b֫̩�A�
*

epsiloniiI�-��W.       ��W�	R�̩�A�
* 

Average reward per stepiiI�A���       ��2	(�̩�A�
*

epsiloniiI��uE.       ��W�	� �̩�A�
* 

Average reward per stepiiI����f       ��2	�!�̩�A�
*

epsiloniiI�-��Z.       ��W�	MJ�̩�A�
* 

Average reward per stepiiI���Π       ��2	+K�̩�A�
*

epsiloniiI����`.       ��W�	p�̩�A�
* 

Average reward per stepiiI��9       ��2	�p�̩�A�
*

epsiloniiI�*�v	.       ��W�	���̩�A�
* 

Average reward per stepiiI�u��_       ��2	���̩�A�
*

epsiloniiI�~
�.       ��W�	� �̩�A�
* 

Average reward per stepiiI��J.&       ��2	��̩�A�
*

epsiloniiI���t.       ��W�	ú̩�A�
* 

Average reward per stepiiI��ca�       ��2	�ú̩�A�
*

epsiloniiI�����.       ��W�	�̩�A�
* 

Average reward per stepiiI����W       ��2	��̩�A�
*

epsiloniiI�g��.       ��W�	Qj�̩�A�
* 

Average reward per stepiiI�S�3       ��2		k�̩�A�
*

epsiloniiI����0       ���_	��̩�A'*#
!
Average reward per episode���>���.       ��W�	ė�̩�A'*!

total reward per episode  �AI~�.       ��W�	Tn�̩�A�
* 

Average reward per step���>bPYW       ��2	/o�̩�A�
*

epsilon���>�"�.       ��W�	ȶ�̩�A�
* 

Average reward per step���>~�       ��2	x��̩�A�
*

epsilon���>��+y.       ��W�	,��̩�A�
* 

Average reward per step���>���<       ��2	���̩�A�
*

epsilon���>�oϛ.       ��W�	��̩�A�
* 

Average reward per step���>��Y�       ��2	T�̩�A�
*

epsilon���>gf�.       ��W�	_}�̩�A�
* 

Average reward per step���>�A       ��2	(~�̩�A�
*

epsilon���>�4).       ��W�	.��̩�A�
* 

Average reward per step���>��dM       ��2	��̩�A�
*

epsilon���>%���.       ��W�	E��̩�A�
* 

Average reward per step���>S���       ��2	��̩�A�
*

epsilon���>�G	�.       ��W�	�C�̩�A�
* 

Average reward per step���>�oO       ��2	pD�̩�A�
*

epsilon���>�%%.       ��W�	DQ�̩�A�
* 

Average reward per step���>14#�       ��2	�Q�̩�A�
*

epsilon���>�9�T.       ��W�	1^�̩�A�
* 

Average reward per step���>({c       ��2	�^�̩�A�
*

epsilon���>���.       ��W�	��̩�A�
* 

Average reward per step���>.R3       ��2	k��̩�A�
*

epsilon���>��~.       ��W�	���̩�A�
* 

Average reward per step���>'&�"       ��2	\��̩�A�
*

epsilon���>'Y�b.       ��W�	���̩�A�
* 

Average reward per step���>�W

       ��2	1��̩�A�
*

epsilon���>�)��.       ��W�	���̩�A�
* 

Average reward per step���>2�@'       ��2	���̩�A�
*

epsilon���>�\�.       ��W�	![�̩�A�
* 

Average reward per step���>t(v�       ��2	�[�̩�A�
*

epsilon���>�R`�.       ��W�	���̩�A�
* 

Average reward per step���>�-       ��2	���̩�A�
*

epsilon���>�z�.       ��W�	�̩�A�
* 

Average reward per step���>%g�       ��2	��̩�A�
*

epsilon���>�`.       ��W�	�̩�A�
* 

Average reward per step���>M+�       ��2	��̩�A�
*

epsilon���>��.       ��W�	Ǆ�̩�A�
* 

Average reward per step���>f�g       ��2	υ�̩�A�
*

epsilon���>�镞.       ��W�	���̩�A�
* 

Average reward per step���>H
3       ��2	|��̩�A�
*

epsilon���>�i��.       ��W�	���̩�A�
* 

Average reward per step���>���       ��2	���̩�A�
*

epsilon���>���.       ��W�	��̩�A�
* 

Average reward per step���>���       ��2	���̩�A�
*

epsilon���>�U�.       ��W�	���̩�A�
* 

Average reward per step���>;���       ��2	���̩�A�
*

epsilon���>�?��0       ���_	��̩�A(*#
!
Average reward per episodez�����ܧ.       ��W�	��̩�A(*!

total reward per episode  �ƌMr.       ��W�	��̩�A�
* 

Average reward per stepz���+��       ��2	���̩�A�
*

epsilonz����yQ+.       ��W�	���̩�A�
* 

Average reward per stepz���KM�,       ��2	���̩�A�
*

epsilonz���fE�.       ��W�	-w�̩�A�
* 

Average reward per stepz����T�       ��2	�w�̩�A�
*

epsilonz���8B�.       ��W�	ӿ�̩�A�
* 

Average reward per stepz���O`Ȣ       ��2	���̩�A�
*

epsilonz���)wPL.       ��W�	��̩�A�
* 

Average reward per stepz���*q        ��2	��̩�A�
*

epsilonz�����.       ��W�	�"̩�A�
* 

Average reward per stepz����Tؘ       ��2	l#̩�A�
*

epsilonz�����'~.       ��W�	x�̩�A�
* 

Average reward per stepz������+       ��2	R�̩�A�
*

epsilonz�����.       ��W�	,�̩�A�
* 

Average reward per stepz���d�       ��2	��̩�A�
*

epsilonz����-�p.       ��W�	v
̩�A�
* 

Average reward per stepz���I��       ��2	&
̩�A�
*

epsilonz������d.       ��W�	<K̩�A�
* 

Average reward per stepz������\       ��2	�K̩�A�
*

epsilonz�����(.       ��W�	X�̩�A�
* 

Average reward per stepz�����O�       ��2	2�̩�A�
*

epsilonz���sκ.       ��W�	 )̩�A�
* 

Average reward per stepz�������       ��2	�)̩�A�
*

epsilonz���RDYQ.       ��W�	y;̩�A�
* 

Average reward per stepz����]�c       ��2	2<̩�A�
*

epsilonz������Z.       ��W�	�̩�A�
* 

Average reward per stepz���C��       ��2	�̩�A�
*

epsilonz���J�4g.       ��W�	U/̩�A�
* 

Average reward per stepz�����jK       ��2	M0̩�A�
*

epsilonz�����B�.       ��W�	R�̩�A�
* 

Average reward per stepz����8�2       ��2	�̩�A�
*

epsilonz���?~D.       ��W�	�̩�A�
* 

Average reward per stepz���Q��       ��2	�̩�A�
*

epsilonz���HL/.       ��W�	�m̩�A�
* 

Average reward per stepz�������       ��2	�n̩�A�
*

epsilonz���|��m.       ��W�	��̩�A�
* 

Average reward per stepz������       ��2	l�̩�A�
*

epsilonz���X�9�.       ��W�	<�̩�A�
* 

Average reward per stepz����-c       ��2	0�̩�A�
*

epsilonz���r?a�.       ��W�	�`!̩�A�
* 

Average reward per stepz���ܼ~�       ��2	_a!̩�A�
*

epsilonz���z4��.       ��W�	U�#̩�A�
* 

Average reward per stepz����4�8       ��2	��#̩�A�
*

epsilonz����1�.       ��W�	I�%̩�A�
* 

Average reward per stepz����_a�       ��2	�%̩�A�
*

epsilonz����7Q�.       ��W�	��'̩�A�
* 

Average reward per stepz���UI��       ��2	O�'̩�A�
*

epsilonz�����60       ���_	G(̩�A)*#
!
Average reward per episodeUU��mȥe.       ��W�	�(̩�A)*!

total reward per episode  )���|].       ��W�	��-̩�A�
* 

Average reward per stepUU����m       ��2	T�-̩�A�
*

epsilonUU���:�.       ��W�	z0̩�A�
* 

Average reward per stepUU����z�       ��2	T0̩�A�
*

epsilonUU���G�.       ��W�	742̩�A�
* 

Average reward per stepUU��`$i|       ��2	52̩�A�
*

epsilonUU����{[.       ��W�	#�4̩�A�
* 

Average reward per stepUU��8Z�n       ��2	�4̩�A�
*

epsilonUU���=�.       ��W�	�a8̩�A�
* 

Average reward per stepUU��q��       ��2	�b8̩�A�
*

epsilonUU�����.       ��W�	N�:̩�A�
* 

Average reward per stepUU���hD7       ��2	��:̩�A�
*

epsilonUU���	r.       ��W�	W�<̩�A�
* 

Average reward per stepUU��ۍ�       ��2	-�<̩�A�
*

epsilonUU������.       ��W�	>̩�A�
* 

Average reward per stepUU��6q>�       ��2	�>̩�A�
*

epsilonUU��C�VQ.       ��W�	�d@̩�A�
* 

Average reward per stepUU���f�!       ��2	�e@̩�A�
*

epsilonUU����S.       ��W�	�B̩�A�
* 

Average reward per stepUU����LQ       ��2	��B̩�A�
*

epsilonUU��u5�.       ��W�	5�D̩�A�
* 

Average reward per stepUU��/6`K       ��2	КD̩�A�
*

epsilonUU��F��W.       ��W�	��F̩�A�
* 

Average reward per stepUU����Y�       ��2	��F̩�A�
*

epsilonUU���F1.       ��W�	�H̩�A�
* 

Average reward per stepUU������       ��2	��H̩�A�
*

epsilonUU���6i�.       ��W�	��J̩�A�
* 

Average reward per stepUU���5��       ��2	e K̩�A�
*

epsilonUU�����t.       ��W�	?XN̩�A�
* 

Average reward per stepUU���H�E       ��2	>YN̩�A�
*

epsilonUU��?:l�.       ��W�	7�P̩�A�
* 

Average reward per stepUU��H�#�       ��2	��P̩�A�
*

epsilonUU���f|�.       ��W�	��R̩�A�
* 

Average reward per stepUU������       ��2	o�R̩�A�
*

epsilonUU��n��0       ���_	�S̩�A**#
!
Average reward per episode���+.       ��W�	iS̩�A**!

total reward per episode  (�*���.       ��W�	��V̩�A�
* 

Average reward per step�fF<�       ��2	F�V̩�A�
*

epsilon�լb.       ��W�	�+Y̩�A�
* 

Average reward per step��P%       ��2	�,Y̩�A�
*

epsilon��?�.       ��W�	bH[̩�A�
* 

Average reward per step�O��       ��2	I[̩�A�
*

epsilon�O4��.       ��W�	��]̩�A�
* 

Average reward per step���X       ��2	I�]̩�A�
*

epsilon��&7�.       ��W�	C�_̩�A�
* 

Average reward per step��\�       ��2	��_̩�A�
*

epsilon�-��Z.       ��W�	�b̩�A�
* 

Average reward per step��U�       ��2	�b̩�A�
*

epsilon�J�Q.       ��W�	L�c̩�A�
* 

Average reward per step�n���       ��2	�c̩�A�
*

epsilon�-�L�.       ��W�	�8f̩�A�
* 

Average reward per step�w�I�       ��2	z9f̩�A�
*

epsilon�,Ԣ�.       ��W�	v�g̩�A�
* 

Average reward per step�M�,       ��2	G�g̩�A�
*

epsilon�z�'.       ��W�	�i̩�A�
* 

Average reward per step��f��       ��2	Mi̩�A�
*

epsilon�_"��.       ��W�	� k̩�A�
* 

Average reward per step��N       ��2	�!k̩�A�
*

epsilon�n�y�.       ��W�	8m̩�A�
* 

Average reward per step�w�2P       ��2	9m̩�A�
*

epsilon�U.       ��W�	M�o̩�A�
* 

Average reward per step��;�       ��2	<�o̩�A�
*

epsilon���.       ��W�	�as̩�A�
* 

Average reward per step�����       ��2	Ebs̩�A�
*

epsilon��c�2.       ��W�	�	u̩�A�
* 

Average reward per step�\�ҋ       ��2	�
u̩�A�
*

epsilon�H�3y.       ��W�	�3w̩�A�
* 

Average reward per step���R�       ��2	r4w̩�A�
*

epsilon�����.       ��W�	�Yy̩�A�
* 

Average reward per step�{?;�       ��2	�Zy̩�A�
*

epsilon�H��.       ��W�	/k{̩�A�
* 

Average reward per step��΂]       ��2	�k{̩�A�
*

epsilon�<��.       ��W�		�}̩�A�
* 

Average reward per step�T�k       ��2	ߊ}̩�A�
*

epsilon�ⴔ\.       ��W�	��̩�A�
* 

Average reward per step����       ��2	�̩�A�
*

epsilon�@'�.       ��W�	���̩�A�
* 

Average reward per step���{<       ��2	B��̩�A�
*

epsilon����.       ��W�	氃̩�A�
* 

Average reward per step���	       ��2	���̩�A�
*

epsilon�O��.       ��W�	��̩�A�
* 

Average reward per step�0z��       ��2	��̩�A�
*

epsilon���,.       ��W�	=|�̩�A�
* 

Average reward per step�qz�7       ��2	9}�̩�A�
*

epsilon�tj�m.       ��W�	7��̩�A�
* 

Average reward per step���^       ��2	ک�̩�A�
*

epsilon���2�.       ��W�	�͋̩�A�
* 

Average reward per step��;��       ��2	d΋̩�A�
*

epsilon��A�9.       ��W�	��̩�A�
* 

Average reward per step�wC�       ��2	s��̩�A�
*

epsilon�H��.       ��W�	6 �̩�A�
* 

Average reward per step���       ��2	O!�̩�A�
*

epsilon����.       ��W�	�̩�A�
* 

Average reward per step�sEϝ       ��2	PÑ̩�A�
*

epsilon�#�M.       ��W�	fؓ̩�A�
* 

Average reward per step����       ��2	Eٓ̩�A�
*

epsilon�4co*.       ��W�	�̩�A�
* 

Average reward per step��w3�       ��2	��̩�A�
*

epsilon��g��.       ��W�	x%�̩�A�
* 

Average reward per step��	�k       ��2	-&�̩�A�
*

epsilon�C���.       ��W�	���̩�A�
* 

Average reward per step�T�h�       ��2	K �̩�A�
*

epsilon��E]�.       ��W�	6u�̩�A�
* 

Average reward per step���r&       ��2	�u�̩�A�
*

epsilon�S8�.       ��W�	���̩�A�
* 

Average reward per step�����       ��2	V��̩�A�
*

epsilon���p.       ��W�	��̩�A�
* 

Average reward per step�Q2@�       ��2	��̩�A�
*

epsilon�2J(".       ��W�	E�̩�A�
* 

Average reward per step����       ��2	A�̩�A�
*

epsilon�b���.       ��W�	S$�̩�A�
* 

Average reward per step�����       ��2	�$�̩�A�
*

epsilon�X��.       ��W�	���̩�A�
* 

Average reward per step��p��       ��2	Z��̩�A�
*

epsilon�M��.       ��W�	ʬ̩�A�
* 

Average reward per step�S�Z       ��2	�ʬ̩�A�
*

epsilon����.       ��W�	X�̩�A�
* 

Average reward per step�ud��       ��2	!�̩�A�
*

epsilon�f3�.       ��W�	��̩�A�
* 

Average reward per step��8�Y       ��2	8�̩�A�
*

epsilon�*=�.       ��W�	��̩�A�
* 

Average reward per step�����       ��2	� �̩�A�
*

epsilon��2�l.       ��W�	�*�̩�A�
* 

Average reward per step��w��       ��2	A+�̩�A�
*

epsilon��M*U.       ��W�	2��̩�A�* 

Average reward per step�%��A       ��2	6��̩�A�*

epsilon�4*�.       ��W�	��̩�A�* 

Average reward per step�� k       ��2	��̩�A�*

epsilon�*J�7.       ��W�	��̩�A�* 

Average reward per step�/�F_       ��2	��̩�A�*

epsilon��+�.       ��W�	�>�̩�A�* 

Average reward per step��~Q       ��2	h?�̩�A�*

epsilon��RB\.       ��W�	�̩�A�* 

Average reward per step��%�       ��2	ǂ�̩�A�*

epsilon�'5i.       ��W�	`��̩�A�* 

Average reward per step�h��       ��2	��̩�A�*

epsilon��{{.       ��W�	T�̩�A�* 

Average reward per step��o�-       ��2	��̩�A�*

epsilon���.       ��W�	��̩�A�* 

Average reward per step�?%       ��2	���̩�A�*

epsilon����G.       ��W�	t��̩�A�* 

Average reward per step��糘       ��2	R��̩�A�*

epsilon����.       ��W�	{J�̩�A�* 

Average reward per step��1��       ��2	EK�̩�A�*

epsilon�ۄ��.       ��W�	ղ�̩�A�* 

Average reward per step�! k�       ��2	ٳ�̩�A�*

epsilon�3�x.       ��W�	a��̩�A�* 

Average reward per step���J       ��2	  �̩�A�*

epsilon��s�:.       ��W�	 o�̩�A�* 

Average reward per step���;       ��2	�o�̩�A�*

epsilon����E.       ��W�	���̩�A�* 

Average reward per step�� �       ��2	���̩�A�*

epsilon����.       ��W�	�l�̩�A�* 

Average reward per step��#v       ��2	�m�̩�A�*

epsilon�X�O�.       ��W�	���̩�A�* 

Average reward per step�m���       ��2	[��̩�A�*

epsilon�r0��.       ��W�	�w�̩�A�* 

Average reward per step��E��       ��2	�x�̩�A�*

epsilon��f�.       ��W�	��̩�A�* 

Average reward per step����       ��2	c��̩�A�*

epsilon���`.       ��W�	��̩�A�* 

Average reward per step�����       ��2	� �̩�A�*

epsilon��@�.       ��W�	#�̩�A�* 

Average reward per step�,��&       ��2	D�̩�A�*

epsilon��	��.       ��W�	�y�̩�A�* 

Average reward per step���w       ��2	�z�̩�A�*

epsilon�(G��.       ��W�	�2�̩�A�* 

Average reward per step���j�       ��2	74�̩�A�*

epsilon�-��(.       ��W�	���̩�A�* 

Average reward per step�A�4       ��2	���̩�A�*

epsilon���.       ��W�	t��̩�A�* 

Average reward per step�1� �       ��2	���̩�A�*

epsilon��F��.       ��W�	1��̩�A�* 

Average reward per step�{آ�       ��2	t��̩�A�*

epsilon���l;.       ��W�	���̩�A�* 

Average reward per step�j��       ��2	͑�̩�A�*

epsilon�uxfG.       ��W�	p
�̩�A�* 

Average reward per step��<�       ��2	��̩�A�*

epsilon���.       ��W�	���̩�A�* 

Average reward per step�T�KB       ��2	���̩�A�*

epsilon��Ҥ.       ��W�	|��̩�A�* 

Average reward per step����I       ��2	���̩�A�*

epsilon��/�.       ��W�	~U�̩�A�* 

Average reward per step��~s       ��2	�V�̩�A�*

epsilon���YO.       ��W�	3�̩�A�* 

Average reward per step�ن       ��2	�̩�A�*

epsilon��8��.       ��W�	N��̩�A�* 

Average reward per step��L}�       ��2	A��̩�A�*

epsilon�7�>�.       ��W�	���̩�A�* 

Average reward per step�&7&       ��2	x��̩�A�*

epsilon�_}.       ��W�	�̩�A�* 

Average reward per step�߁3+       ��2	�̩�A�*

epsilon��8#�.       ��W�	=~̩�A�* 

Average reward per step����h       ��2	|̩�A�*

epsilon�D>t0.       ��W�	��̩�A�* 

Average reward per step�ٗ��       ��2	��̩�A�*

epsilon��Q�%.       ��W�	Ƥ	̩�A�* 

Average reward per step�[k��       ��2	��	̩�A�*

epsilon�tc �.       ��W�	�̩�A�* 

Average reward per step��p��       ��2	�̩�A�*

epsilon��u�H.       ��W�	��̩�A�* 

Average reward per step�	��       ��2	?�̩�A�*

epsilon����.       ��W�	s�̩�A�* 

Average reward per step��J�       ��2	��̩�A�*

epsilon�˸%.       ��W�	�-̩�A�* 

Average reward per step�I�|�       ��2	�.̩�A�*

epsilon�^�k.       ��W�	��̩�A�* 

Average reward per step���W�       ��2	��̩�A�*

epsilon��9�.       ��W�	�̩�A�* 

Average reward per step���P�       ��2	�̩�A�*

epsilon����0       ���_	�Q̩�A+*#
!
Average reward per episode�f�=]Ҳ.       ��W�	7S̩�A+*!

total reward per episode   A�d��.       ��W�	M� ̩�A�* 

Average reward per step�f�=�x3       ��2	8� ̩�A�*

epsilon�f�=+,�B.       ��W�	��"̩�A�* 

Average reward per step�f�=���       ��2	��"̩�A�*

epsilon�f�=|.       ��W�	g�$̩�A�* 

Average reward per step�f�=�>�       ��2	��$̩�A�*

epsilon�f�=�84H.       ��W�	uW'̩�A�* 

Average reward per step�f�=�g�       ��2	hX'̩�A�*

epsilon�f�=���".       ��W�	C8+̩�A�* 

Average reward per step�f�=%�       ��2	�9+̩�A�*

epsilon�f�=v}.       ��W�	�5/̩�A�* 

Average reward per step�f�=�~�       ��2	�6/̩�A�*

epsilon�f�=]�N�.       ��W�	��3̩�A�* 

Average reward per step�f�=	O�       ��2	��3̩�A�*

epsilon�f�=��]�.       ��W�	'i7̩�A�* 

Average reward per step�f�=��+Y       ��2	�i7̩�A�*

epsilon�f�=%��.       ��W�	�/;̩�A�* 

Average reward per step�f�=݈q       ��2	�0;̩�A�*

epsilon�f�=�ܠ�.       ��W�	i�=̩�A�* 

Average reward per step�f�=��o       ��2	��=̩�A�*

epsilon�f�=�2h�.       ��W�	C@̩�A�* 

Average reward per step�f�=��g       ��2	ND@̩�A�*

epsilon�f�=d�X�.       ��W�	9|C̩�A�* 

Average reward per step�f�=�X�.       ��2	[}C̩�A�*

epsilon�f�=�K.       ��W�	'�F̩�A�* 

Average reward per step�f�=v�       ��2	#�F̩�A�*

epsilon�f�=y~1�.       ��W�	��I̩�A�* 

Average reward per step�f�=�r��       ��2	��I̩�A�*

epsilon�f�=i�!�.       ��W�	S[L̩�A�* 

Average reward per step�f�=ny�       ��2	}\L̩�A�*

epsilon�f�=�r��.       ��W�	
O̩�A�* 

Average reward per step�f�=���       ��2	9O̩�A�*

epsilon�f�=�,	.       ��W�	GR̩�A�* 

Average reward per step�f�=L���       ��2	�R̩�A�*

epsilon�f�=�`��.       ��W�	�tT̩�A�* 

Average reward per step�f�=�Z��       ��2	�uT̩�A�*

epsilon�f�=��*?.       ��W�	�V̩�A�* 

Average reward per step�f�=SF       ��2	u�V̩�A�*

epsilon�f�=�E�0       ���_	�QW̩�A,*#
!
Average reward per episodey��0(.       ��W�	 SW̩�A,*!

total reward per episode  æ�3�.       ��W�	�\̩�A�* 

Average reward per stepy�G���       ��2	a�\̩�A�*

epsilony�@�V:.       ��W�	s.`̩�A�* 

Average reward per stepy��)�?       ��2	�/`̩�A�*

epsilony�&���.       ��W�	�b̩�A�* 

Average reward per stepy�y�       ��2	֌b̩�A�*

epsilony�=��.       ��W�	Z�d̩�A�* 

Average reward per stepy� O��       ��2	<�d̩�A�*

epsilony�*�"�.       ��W�	��f̩�A�* 

Average reward per stepy�.��       ��2	��f̩�A�*

epsilony�hEt|.       ��W�	�Wi̩�A�* 

Average reward per stepy���>       ��2	�Xi̩�A�*

epsilony��[-.       ��W�	m̩�A�* 

Average reward per stepy�h]��       ��2	m̩�A�*

epsilony��:&�.       ��W�	��n̩�A�* 

Average reward per stepy�?<�       ��2	o�n̩�A�*

epsilony��..       ��W�	Z�q̩�A�* 

Average reward per stepy��h       ��2	5�q̩�A�*

epsilony���C:.       ��W�	b�u̩�A�* 

Average reward per stepy��#E�       ��2	8�u̩�A�*

epsilony����.       ��W�	:w̩�A�* 

Average reward per stepy�}l�       ��2	C;w̩�A�*

epsilony��~9�.       ��W�	��y̩�A�* 

Average reward per stepy�s��>       ��2	��y̩�A�*

epsilony�O�Z�.       ��W�	Ef{̩�A�* 

Average reward per stepy�l�       ��2	{g{̩�A�*

epsilony��~u�.       ��W�	-[}̩�A�* 

Average reward per stepy�F Fa       ��2	)\}̩�A�*

epsilony����.       ��W�	��̩�A�* 

Average reward per stepy��[е       ��2	@�̩�A�*

epsilony�\+Ct.       ��W�	@��̩�A�* 

Average reward per stepy��Ym       ��2	#��̩�A�*

epsilony�b��.       ��W�	HŃ̩�A�* 

Average reward per stepy�ab�       ��2	aƃ̩�A�*

epsilony�?6��.       ��W�	��̩�A�* 

Average reward per stepy�BO�b       ��2	�̩�A�*

epsilony����r.       ��W�	�އ̩�A�* 

Average reward per stepy�`�3       ��2	�߇̩�A�*

epsilony��!K�.       ��W�	!\�̩�A�* 

Average reward per stepy����       ��2	_]�̩�A�*

epsilony�ɀ�0.       ��W�	��̩�A�* 

Average reward per stepy��l�       ��2	��̩�A�*

epsilony��SZD.       ��W�	B�̩�A�* 

Average reward per stepy�wv�       ��2	W�̩�A�*

epsilony��qT<.       ��W�	�5�̩�A�* 

Average reward per stepy��w�       ��2	�6�̩�A�*

epsilony����.       ��W�	v�̩�A�* 

Average reward per stepy���L5       ��2	��̩�A�*

epsilony���`.       ��W�	��̩�A�* 

Average reward per stepy���@�       ��2	��̩�A�*

epsilony�����.       ��W�	��̩�A�* 

Average reward per stepy��`ܷ       ��2	B��̩�A�*

epsilony��_�t.       ��W�	V�̩�A�* 

Average reward per stepy��W��       ��2	�W�̩�A�*

epsilony����$.       ��W�	�m�̩�A�* 

Average reward per stepy����r       ��2	�n�̩�A�*

epsilony��u��.       ��W�	��̩�A�* 

Average reward per stepy�����       ��2	���̩�A�*

epsilony��8�y.       ��W�	.;�̩�A�* 

Average reward per stepy�)<�       ��2	G<�̩�A�*

epsilony�1�50       ���_	p�̩�A-*#
!
Average reward per episode����&�+�.       ��W�	2q�̩�A-*!

total reward per episode  ���V{.       ��W�	Mf�̩�A�* 

Average reward per step�����إ       ��2	�g�̩�A�*

epsilon�������.       ��W�	|+�̩�A�* 

Average reward per step�����$٨       ��2	�,�̩�A�*

epsilon����v��F.       ��W�	���̩�A�* 

Average reward per step����v��}       ��2	��̩�A�*

epsilon����q��A.       ��W�	�I�̩�A�* 

Average reward per step����F       ��2	�J�̩�A�*

epsilon����Ƨ�d.       ��W�	cյ̩�A�* 

Average reward per step����s}�       ��2	kֵ̩�A�*

epsilon�����>H.       ��W�	ĕ�̩�A�* 

Average reward per step����i�Ρ       ��2	̖�̩�A�*

epsilon�����"�.       ��W�	`��̩�A�* 

Average reward per step�����_�T       ��2	���̩�A�*

epsilon������2".       ��W�	F�̩�A�* 

Average reward per step����@�C       ��2	�F�̩�A�*

epsilon��������.       ��W�	�;�̩�A�* 

Average reward per step����"P��       ��2	�<�̩�A�*

epsilon������ �.       ��W�	�̩�A�* 

Average reward per step����HMSJ       ��2	7�̩�A�*

epsilon������7�.       ��W�	n��̩�A�* 

Average reward per step������г       ��2	���̩�A�*

epsilon����O�+�.       ��W�	~��̩�A�* 

Average reward per step������P�       ��2	���̩�A�*

epsilon����㝵�.       ��W�	%<�̩�A�* 

Average reward per step�����-YY       ��2	=�̩�A�*

epsilon������$.       ��W�	���̩�A�* 

Average reward per step����S`       ��2	���̩�A�*

epsilon����''��.       ��W�	��̩�A�* 

Average reward per step����%E]�       ��2	߇�̩�A�*

epsilon�����"�[.       ��W�	D��̩�A�* 

Average reward per step����gѐ�       ��2	L��̩�A�*

epsilon�����$�.       ��W�	���̩�A�* 

Average reward per step������ 0       ��2		��̩�A�*

epsilon������q�.       ��W�	���̩�A�* 

Average reward per step�����=9�       ��2	=��̩�A�*

epsilon����|�G.       ��W�	��̩�A�* 

Average reward per step��������       ��2	��̩�A�*

epsilon����`��.       ��W�	�E�̩�A�* 

Average reward per step������A?       ��2	$G�̩�A�*

epsilon����ܱo�.       ��W�	��̩�A�* 

Average reward per step�����Z�+       ��2	7��̩�A�*

epsilon����UI��.       ��W�	M��̩�A�* 

Average reward per step�������        ��2	Y��̩�A�*

epsilon�����4$.       ��W�	��̩�A�* 

Average reward per step������л       ��2	���̩�A�*

epsilon����(v31.       ��W�	� �̩�A�* 

Average reward per step�����(�T       ��2	�!�̩�A�*

epsilon����L B.       ��W�	]��̩�A�* 

Average reward per step������4�       ��2	v��̩�A�*

epsilon�������.       ��W�	���̩�A�* 

Average reward per step����2]U       ��2	֩�̩�A�*

epsilon������.       ��W�	Rb�̩�A�* 

Average reward per step������|       ��2	=c�̩�A�*

epsilon������1v.       ��W�	/�̩�A�* 

Average reward per step�����`5       ��2	Y�̩�A�*

epsilon����2�(�.       ��W�	���̩�A�* 

Average reward per step����O��z       ��2	���̩�A�*

epsilon����*�Go.       ��W�	>\̩�A�* 

Average reward per step�����[u       ��2	1]̩�A�*

epsilon����x���.       ��W�	f�̩�A�* 

Average reward per step����6��0       ��2	D�̩�A�*

epsilon����5V�k.       ��W�	�x̩�A�* 

Average reward per step����$�
~       ��2	�y̩�A�*

epsilon������d.       ��W�	�`̩�A�* 

Average reward per step����D���       ��2	[a̩�A�*

epsilon����:}z$.       ��W�	�̩�A�* 

Average reward per step����ӈ��       ��2	�̩�A�*

epsilon����z��c.       ��W�	�_̩�A�* 

Average reward per step����s�M�       ��2	�`̩�A�*

epsilon�������.       ��W�	l�̩�A�* 

Average reward per step������"       ��2	��̩�A�*

epsilon��������.       ��W�	�|̩�A�* 

Average reward per step�����\�P       ��2	�}̩�A�*

epsilon����Y8.       ��W�	x~̩�A�* 

Average reward per step����G_�       ��2	Z̩�A�*

epsilon������2.       ��W�	��̩�A�* 

Average reward per step����� �       ��2	��̩�A�*

epsilon�����*�.       ��W�	��̩�A�* 

Average reward per step������?       ��2	�̩�A�*

epsilon����F�.       ��W�	ܡ̩�A�* 

Average reward per step����`�B       ��2	�̩�A�*

epsilon����j=t�.       ��W�	�"!̩�A�* 

Average reward per step�������9       ��2	�#!̩�A�*

epsilon�����(�.       ��W�	Wz#̩�A�* 

Average reward per step����c!<�       ��2	�{#̩�A�*

epsilon����`*.       ��W�	��&̩�A�* 

Average reward per step����G�e�       ��2	k�&̩�A�*

epsilon����<�$.       ��W�	C)̩�A�* 

Average reward per step�����H��       ��2	5D)̩�A�*

epsilon�����"�`.       ��W�	�e+̩�A�* 

Average reward per step����g�Ě       ��2	�f+̩�A�*

epsilon����C_'u.       ��W�	F�-̩�A�* 

Average reward per step����9��       ��2	��-̩�A�*

epsilon�����#�.       ��W�	�t1̩�A�* 

Average reward per step����\���       ��2	�u1̩�A�*

epsilon����H�\�.       ��W�	JE3̩�A�* 

Average reward per step�����Vӿ       ��2	,F3̩�A�*

epsilon����
A��.       ��W�	Z�5̩�A�* 

Average reward per step������|L       ��2	,�5̩�A�*

epsilon����6C�[.       ��W�	�8̩�A�* 

Average reward per step����M��       ��2	�8̩�A�*

epsilon���� �4.       ��W�	��;̩�A�* 

Average reward per step����ܾT;       ��2	��;̩�A�*

epsilon����.�.       ��W�	�@>̩�A�* 

Average reward per step����s	�+       ��2	B>̩�A�*

epsilon����{��.       ��W�	�'B̩�A�* 

Average reward per step�����n�       ��2	()B̩�A�*

epsilon�����@�X.       ��W�	��F̩�A�* 

Average reward per step������W#       ��2	��F̩�A�*

epsilon������6.       ��W�	��J̩�A�* 

Average reward per step����(�.       ��2	u�J̩�A�*

epsilon����M�O.       ��W�	'N̩�A�* 

Average reward per step����w�o�       ��2	'N̩�A�*

epsilon�����1s.       ��W�	��O̩�A�* 

Average reward per step�����T(�       ��2	��O̩�A�*

epsilon�����I̢.       ��W�	YmR̩�A�* 

Average reward per step����lQ1�       ��2	znR̩�A�*

epsilon����}.-I.       ��W�	�$T̩�A�* 

Average reward per step����pqgv       ��2	�%T̩�A�*

epsilon����D)��.       ��W�	H�V̩�A�* 

Average reward per step�����J�       ��2	r�V̩�A�*

epsilon�������n.       ��W�	(Z̩�A�* 

Average reward per step�����r6       ��2	�(Z̩�A�*

epsilon������5�.       ��W�	9�\̩�A�* 

Average reward per step����4�G       ��2	B�\̩�A�*

epsilon�����9e.       ��W�	�l^̩�A�* 

Average reward per step����#ڕ!       ��2	�m^̩�A�*

epsilon����L���.       ��W�	2>a̩�A�* 

Average reward per step��������       ��2	?a̩�A�*

epsilon����f�Ѕ.       ��W�	��d̩�A�* 

Average reward per step���� I?|       ��2	��d̩�A�*

epsilon����&�.       ��W�	�Vg̩�A�* 

Average reward per step����4���       ��2	�Wg̩�A�*

epsilon����dqT'.       ��W�	�*k̩�A�* 

Average reward per step����6iZ�       ��2	�+k̩�A�*

epsilon�����!L.       ��W�	r�m̩�A�* 

Average reward per step�����ZK�       ��2	��m̩�A�*

epsilon�����qx-.       ��W�	#q̩�A�* 

Average reward per step����	�g       ��2	$q̩�A�*

epsilon����n窊.       ��W�	es̩�A�* 

Average reward per step����ꏋ�       ��2	�s̩�A�*

epsilon�����XL.       ��W�	ogu̩�A�* 

Average reward per step����Hr��       ��2	�hu̩�A�*

epsilon����w���.       ��W�	9�w̩�A�* 

Average reward per step�����M�k       ��2	�w̩�A�*

epsilon������JP.       ��W�	�5|̩�A�* 

Average reward per step������B�       ��2	�6|̩�A�*

epsilon�����s�E.       ��W�	y�̩�A�* 

Average reward per step����9��       ��2	��̩�A�*

epsilon����y���.       ��W�	�g�̩�A�* 

Average reward per step����.$��       ��2	�h�̩�A�*

epsilon����J��.       ��W�	$B�̩�A�* 

Average reward per step����6J[�       ��2	=C�̩�A�*

epsilon����iTa.       ��W�	���̩�A�* 

Average reward per step�����ף�       ��2	氈̩�A�*

epsilon����wǬ�.       ��W�	��̩�A�* 

Average reward per step����b�٩       ��2	��̩�A�*

epsilon�����-m.       ��W�	t�̩�A�* 

Average reward per step������8       ��2	�t�̩�A�*

epsilon����v6�.       ��W�	\�̩�A�* 

Average reward per step����tAo       ��2	m�̩�A�*

epsilon����Y���.       ��W�	節̩�A�* 

Average reward per step�����	�       ��2	ɪ�̩�A�*

epsilon����}o3�.       ��W�	���̩�A�* 

Average reward per step����#Ӎ       ��2	���̩�A�*

epsilon����fF.       ��W�	PQ�̩�A�* 

Average reward per step����cb��       ��2	�R�̩�A�*

epsilon�����A�8.       ��W�	�Ŝ̩�A�* 

Average reward per step����v4ͺ       ��2	`ǜ̩�A�*

epsilon����/�3l.       ��W�	_}�̩�A�* 

Average reward per step����R8*�       ��2	t~�̩�A�*

epsilon�����Y.       ��W�	�"�̩�A�* 

Average reward per step�����F�3       ��2	-$�̩�A�*

epsilon����W�`L.       ��W�	&�̩�A�* 

Average reward per step����H��       ��2	9'�̩�A�*

epsilon����J���.       ��W�	���̩�A�* 

Average reward per step����9���       ��2	�̩�A�*

epsilon�����v&.       ��W�	�z�̩�A�* 

Average reward per step����XT6x       ��2	�{�̩�A�*

epsilon������.       ��W�	3ޯ̩�A�* 

Average reward per step������u.       ��2	+߯̩�A�*

epsilon������B.       ��W�	��̩�A�* 

Average reward per step�����D�r       ��2	��̩�A�*

epsilon����!ɿ'.       ��W�	��̩�A�* 

Average reward per step������xq       ��2	ۧ�̩�A�*

epsilon����c� O.       ��W�	���̩�A�* 

Average reward per step�������\       ��2	���̩�A�*

epsilon������/.       ��W�	;�̩�A�* 

Average reward per step����ݰ/�       ��2	&�̩�A�*

epsilon�����R��.       ��W�	5�̩�A�* 

Average reward per step����=6D�       ��2	_�̩�A�*

epsilon����ݦ�.       ��W�	>x�̩�A�* 

Average reward per step������       ��2	�y�̩�A�*

epsilon������.       ��W�	�P�̩�A�* 

Average reward per step������{       ��2	�Q�̩�A�*

epsilon�������B.       ��W�	T��̩�A�* 

Average reward per step����F��       ��2	*��̩�A�*

epsilon����|�B8.       ��W�	�2�̩�A�* 

Average reward per step�������       ��2	�3�̩�A�*

epsilon����_��0       ���_	]�̩�A.*#
!
Average reward per episode����4.       ��W�	�]�̩�A.*!

total reward per episode  P�X�}�.       ��W�	Ym�̩�A�* 

Average reward per step��[�K�       ��2	Yn�̩�A�*

epsilon��l2�.       ��W�	���̩�A�* 

Average reward per step���n�Z       ��2	��̩�A�*

epsilon��w�S.       ��W�	���̩�A�* 

Average reward per step����       ��2	e��̩�A�*

epsilon��)�4�.       ��W�	���̩�A�* 

Average reward per step��܆�}       ��2	?��̩�A�*

epsilon���%ƈ.       ��W�	e��̩�A�* 

Average reward per step�����v       ��2	2��̩�A�*

epsilon��][i�.       ��W�	��̩�A�* 

Average reward per step��/s�       ��2	��̩�A�*

epsilon����Y	.       ��W�	��̩�A�* 

Average reward per step��,UxR       ��2	���̩�A�*

epsilon���S�.       ��W�	���̩�A�* 

Average reward per step���N8&       ��2	���̩�A�*

epsilon����-�.       ��W�	�D�̩�A�* 

Average reward per step�����       ��2	|E�̩�A�*

epsilon��R��.       ��W�	��̩�A�* 

Average reward per step����       ��2	ˁ�̩�A�*

epsilon��Ut��.       ��W�	W��̩�A�* 

Average reward per step��Χ�q       ��2	:��̩�A�*

epsilon��
��.       ��W�	���̩�A�* 

Average reward per step������       ��2	���̩�A�*

epsilon�����.       ��W�	�D�̩�A�* 

Average reward per step���8�       ��2	�E�̩�A�*

epsilon���vA.       ��W�	Ps�̩�A�* 

Average reward per step��n�x�       ��2	�s�̩�A�*

epsilon��0��Z.       ��W�	y��̩�A�* 

Average reward per step��$iɐ       ��2	O��̩�A�*

epsilon��I'��.       ��W�	��̩�A�* 

Average reward per step��֭�,       ��2	χ�̩�A�*

epsilon��5���0       ���_	D��̩�A/*#
!
Average reward per episode  ��8�5.       ��W�	פ�̩�A/*!

total reward per episode  ���.       ��W�	Н�̩�A�* 

Average reward per step  �bU       ��2	���̩�A�*

epsilon  � �t.       ��W�	���̩�A�* 

Average reward per step  �ĝ-       ��2	���̩�A�*

epsilon  ��S�.       ��W�	�� ̩�A�* 

Average reward per step  �
\�       ��2	$� ̩�A�*

epsilon  �b��m.       ��W�	�̩�A�* 

Average reward per step  ��fЄ       ��2	�̩�A�*

epsilon  ���S�.       ��W�	�̩�A�* 

Average reward per step  ����       ��2	�̩�A�*

epsilon  ����.       ��W�	��̩�A�* 

Average reward per step  ��[�        ��2	��̩�A�*

epsilon  ��,��.       ��W�	�	̩�A�* 

Average reward per step  ��kK�       ��2	�	̩�A�*

epsilon  ��ω�.       ��W�	��
̩�A�* 

Average reward per step  �¦�.       ��2	[�
̩�A�*

epsilon  ��}�.       ��W�	0�̩�A�* 

Average reward per step  ��T`�       ��2	Ǻ̩�A�*

epsilon  �Z�.       ��W�	��̩�A�* 

Average reward per step  ��[N,       ��2	��̩�A�*

epsilon  �z��.       ��W�	�$̩�A�* 

Average reward per step  ��HN       ��2	c%̩�A�*

epsilon  �I�.       ��W�	_̩�A�* 

Average reward per step  �ցM�       ��2	�_̩�A�*

epsilon  ��ʼ�.       ��W�	��̩�A�* 

Average reward per step  �ۃ�u       ��2	N�̩�A�*

epsilon  ��U�?.       ��W�	G̩�A�* 

Average reward per step  �o��6       ��2	̩�A�*

epsilon  ��	4�.       ��W�	�X̩�A�* 

Average reward per step  �}��       ��2	�Y̩�A�*

epsilon  �󀟼.       ��W�	�d̩�A�* 

Average reward per step  ���_�       ��2	�e̩�A�*

epsilon  ����.       ��W�	��̩�A�* 

Average reward per step  ���g       ��2	��̩�A�*

epsilon  �wqF.       ��W�	�c̩�A�* 

Average reward per step  �g�W#       ��2	�d̩�A�*

epsilon  ��+t�.       ��W�	Z�!̩�A�* 

Average reward per step  ��l�r       ��2	��!̩�A�*

epsilon  ��H�.       ��W�	��#̩�A�* 

Average reward per step  ����       ��2	��#̩�A�*

epsilon  ��12.       ��W�	w�'̩�A�* 

Average reward per step  ��/�       ��2	��'̩�A�*

epsilon  ��[Nm.       ��W�	A�)̩�A�* 

Average reward per step  ��!��       ��2	��)̩�A�*

epsilon  �	�.%0       ���_	5�)̩�A0*#
!
Average reward per episode/����j-�.       ��W�	��)̩�A0*!

total reward per episode  +�!.�.       ��W�	9�-̩�A�* 

Average reward per step/�����̡       ��2	��-̩�A�*

epsilon/����U �.       ��W�	6�/̩�A�* 

Average reward per step/���}&�9       ��2	�/̩�A�*

epsilon/����f��.       ��W�	d�1̩�A�* 

Average reward per step/����-�       ��2	!�1̩�A�*

epsilon/����x�9.       ��W�	��3̩�A�* 

Average reward per step/���Ԏs�       ��2	�3̩�A�*

epsilon/�����N.       ��W�	
6̩�A�* 

Average reward per step/����)�       ��2	�
6̩�A�*

epsilon/����tq.       ��W�	<O8̩�A�* 

Average reward per step/���ɥ�       ��2	P8̩�A�*

epsilon/���[�|.       ��W�	N�9̩�A�* 

Average reward per step/�����;       ��2	��9̩�A�*

epsilon/���{���.       ��W�	�<̩�A�* 

Average reward per step/������a       ��2	u<̩�A�*

epsilon/����3:.       ��W�	=I>̩�A�* 

Average reward per step/�����X�       ��2	J>̩�A�*

epsilon/���>^��.       ��W�	�z@̩�A�* 

Average reward per step/���Ea�O       ��2	|@̩�A�*

epsilon/����!�|.       ��W�	��A̩�A�* 

Average reward per step/����K�       ��2	Z�A̩�A�*

epsilon/���Tgy�.       ��W�	�D̩�A�* 

Average reward per step/���N�Ɨ       ��2	�D̩�A�*

epsilon/����kǂ.       ��W�	�)F̩�A�* 

Average reward per step/�����}�       ��2	�*F̩�A�*

epsilon/�����si.       ��W�	�H̩�A�* 

Average reward per step/���Lg�w       ��2	��H̩�A�*

epsilon/���?EA�.       ��W�	NDJ̩�A�* 

Average reward per step/���&��Y       ��2	�DJ̩�A�*

epsilon/����S.       ��W�	�RL̩�A�* 

Average reward per step/���8k��       ��2	7SL̩�A�*

epsilon/����Т.       ��W�	��N̩�A�* 

Average reward per step/����       ��2	@�N̩�A�*

epsilon/����U�'.       ��W�	�,P̩�A�* 

Average reward per step/������       ��2	�-P̩�A�*

epsilon/����,�.       ��W�	�yR̩�A�* 

Average reward per step/�����
       ��2	WzR̩�A�*

epsilon/���h��%0       ���_	�R̩�A1*#
!
Average reward per episodey�}�!�.       ��W�	àR̩�A1*!

total reward per episode  �S!c.       ��W�	g�X̩�A�* 

Average reward per stepy�ʾ�e       ��2	5�X̩�A�*

epsilony��l��.       ��W�	��Z̩�A�* 

Average reward per stepy��xPy       ��2	p�Z̩�A�*

epsilony����`.       ��W�	�]̩�A�* 

Average reward per stepy���!I       ��2	@]̩�A�*

epsilony�Iz�>.       ��W�	�%_̩�A�* 

Average reward per stepy�|�^A       ��2	W&_̩�A�*

epsilony�/��@.       ��W�	p�`̩�A�* 

Average reward per stepy���J�       ��2	_�`̩�A�*

epsilony����.       ��W�	�c̩�A�* 

Average reward per stepy��u�O       ��2	�c̩�A�*

epsilony���u�.       ��W�	��d̩�A�* 

Average reward per stepy�z��P       ��2	��d̩�A�*

epsilony�Љ�).       ��W�	��f̩�A�* 

Average reward per stepy�# �B       ��2	��f̩�A�*

epsilony��?i.       ��W�	�i̩�A�* 

Average reward per stepy��1u�       ��2	�i̩�A�*

epsilony�s��/.       ��W�	��j̩�A�* 

Average reward per stepy��L�       ��2	��j̩�A�*

epsilony�(�q.       ��W�	rm̩�A�* 

Average reward per stepy����        ��2	�m̩�A�*

epsilony��kλ.       ��W�	�?o̩�A�* 

Average reward per stepy�īC�       ��2	�@o̩�A�*

epsilony���A\.       ��W�	+hq̩�A�* 

Average reward per stepy��bx�       ��2	iq̩�A�*

epsilony�[�A�.       ��W�	Țs̩�A�* 

Average reward per stepy�gJ�       ��2	��s̩�A�*

epsilony��
+.       ��W�	�u̩�A�* 

Average reward per stepy���&       ��2	�	u̩�A�*

epsilony����.       ��W�	��w̩�A�* 

Average reward per stepy�ݥ�       ��2	=�w̩�A�*

epsilony���$..       ��W�	.p{̩�A�* 

Average reward per stepy��	��       ��2	�p{̩�A�*

epsilony�V��.       ��W�	&�}̩�A�* 

Average reward per stepy�-�J       ��2	ߌ}̩�A�*

epsilony�ٓs.       ��W�	d�̩�A�* 

Average reward per stepy����y       ��2	C�̩�A�*

epsilony�P�<.       ��W�	!΁̩�A�* 

Average reward per stepy����\       ��2	ρ̩�A�*

epsilony�q!a�.       ��W�	��̩�A�* 

Average reward per stepy��_m       ��2	��̩�A�*

epsilony�//��.       ��W�	�a�̩�A�* 

Average reward per stepy�T� ,       ��2	ob�̩�A�*

epsilony��y4.       ��W�	[�̩�A�* 

Average reward per stepy�?�       ��2	!�̩�A�*

epsilony�x�`{.       ��W�	C��̩�A�* 

Average reward per stepy�!#�        ��2	 �̩�A�*

epsilony�?N��.       ��W�	� �̩�A�* 

Average reward per stepy��֭�       ��2	�!�̩�A�*

epsilony�1�!.       ��W�	���̩�A�* 

Average reward per stepy�䮅�       ��2	c��̩�A�*

epsilony��v�.       ��W�	��̩�A�* 

Average reward per stepy�b�       ��2	E�̩�A�*

epsilony����.       ��W�	3�̩�A�* 

Average reward per stepy���7       ��2	��̩�A�*

epsilony��c��.       ��W�	M�̩�A�* 

Average reward per stepy�_+��       ��2	�M�̩�A�*

epsilony�Q #�.       ��W�	a̩�A�* 

Average reward per stepy�S �       ��2	@Õ̩�A�*

epsilony�����.       ��W�	;�̩�A�* 

Average reward per stepy�_�n       ��2	�̩�A�*

epsilony�D�!�.       ��W�	�1�̩�A�* 

Average reward per stepy�Q���       ��2	�2�̩�A�*

epsilony���h.       ��W�	:��̩�A�* 

Average reward per stepy��Q$       ��2	ٯ�̩�A�*

epsilony����e.       ��W�	\�̩�A�* 

Average reward per stepy��h�       ��2	�̩�A�*

epsilony��Ӗ�.       ��W�	T �̩�A�* 

Average reward per stepy��       ��2	�̩�A�*

epsilony�`�y�0       ���_	��̩�A2*#
!
Average reward per episode�_����.       ��W�	��̩�A2*!

total reward per episode  ��sb�q.       ��W�	��̩�A�* 

Average reward per step�_�tm��       ��2	� �̩�A�*

epsilon�_�y:\`.       ��W�	'�̩�A�* 

Average reward per step�_��<�       ��2	�'�̩�A�*

epsilon�_�<���.       ��W�	+L�̩�A�* 

Average reward per step�_��}
v       ��2	�L�̩�A�*

epsilon�_��K��.       ��W�	;��̩�A�* 

Average reward per step�_�'k7       ��2	��̩�A�*

epsilon�_���I�.       ��W�	f�̩�A�* 

Average reward per step�_�@��       ��2	,�̩�A�*

epsilon�_�	�?�.       ��W�	��̩�A�* 

Average reward per step�_��4�o       ��2	ǡ�̩�A�*

epsilon�_�8�J<.       ��W�	v8�̩�A�* 

Average reward per step�_��G[C       ��2	79�̩�A�*

epsilon�_�� >.       ��W�	�b�̩�A�* 

Average reward per step�_���e       ��2	^c�̩�A�*

epsilon�_�$q$.       ��W�	�V�̩�A�* 

Average reward per step�_�2       ��2	�W�̩�A�*

epsilon�_���c.       ��W�	�׺̩�A�* 

Average reward per step�_��U:       ��2	�غ̩�A�*

epsilon�_�!�U4.       ��W�	�p�̩�A�* 

Average reward per step�_�o�h�       ��2	eq�̩�A�*

epsilon�_����m.       ��W�	+K�̩�A�* 

Average reward per step�_���8�       ��2	+L�̩�A�*

epsilon�_�gd��.       ��W�	�v�̩�A�* 

Average reward per step�_��$]       ��2	�w�̩�A�*

epsilon�_�K�.       ��W�	F�̩�A�* 

Average reward per step�_���G�       ��2	AH�̩�A�*

epsilon�_�ɶ��.       ��W�	4,�̩�A�* 

Average reward per step�_��U��       ��2	-�̩�A�*

epsilon�_�3
��.       ��W�	�P�̩�A�* 

Average reward per step�_�BUm       ��2	iQ�̩�A�*

epsilon�_�2ز.       ��W�	5a�̩�A�* 

Average reward per step�_��c��       ��2	b�̩�A�*

epsilon�_���II.       ��W�	_��̩�A�* 

Average reward per step�_�%t,       ��2	_��̩�A�*

epsilon�_�ρ�j.       ��W�	|��̩�A�* 

Average reward per step�_�C��N       ��2	b��̩�A�*

epsilon�_���.       ��W�	{�̩�A�* 

Average reward per step�_�!Cل       ��2	9|�̩�A�*

epsilon�_�h��f.       ��W�	͒�̩�A�* 

Average reward per step�_���4i       ��2	���̩�A�*

epsilon�_��4;�.       ��W�	Ժ�̩�A�* 

Average reward per step�_�M[A       ��2	���̩�A�*

epsilon�_�>e�.       ��W�	�#�̩�A�* 

Average reward per step�_���*       ��2	�$�̩�A�*

epsilon�_�pV �.       ��W�	z��̩�A�* 

Average reward per step�_���`       ��2	q��̩�A�*

epsilon�_��;L�.       ��W�		�̩�A�* 

Average reward per step�_��̥o       ��2	�	�̩�A�*

epsilon�_����Q.       ��W�	ڬ�̩�A�* 

Average reward per step�_��f�       ��2	��̩�A�*

epsilon�_�T�R�.       ��W�	]��̩�A�* 

Average reward per step�_�;a�       ��2	���̩�A�*

epsilon�_���L6.       ��W�	�R�̩�A�* 

Average reward per step�_��":       ��2	;U�̩�A�*

epsilon�_�0�Ѓ.       ��W�	1%�̩�A�* 

Average reward per step�_�`Y��       ��2	&�̩�A�*

epsilon�_�ԩB>.       ��W�	���̩�A�* 

Average reward per step�_�J���       ��2	R��̩�A�*

epsilon�_��v��.       ��W�	D�̩�A�* 

Average reward per step�_�*��7       ��2	�̩�A�*

epsilon�_�B�6f.       ��W�	�:�̩�A�* 

Average reward per step�_��[D�       ��2	�;�̩�A�*

epsilon�_�=�a.       ��W�	���̩�A�* 

Average reward per step�_�����       ��2	���̩�A�*

epsilon�_�Mͨ.       ��W�	��̩�A�* 

Average reward per step�_����E       ��2	��̩�A�*

epsilon�_��8H�.       ��W�	f��̩�A�* 

Average reward per step�_����       ��2	��̩�A�*

epsilon�_�6y��0       ���_	'��̩�A3*#
!
Average reward per episode۶m��_@.       ��W�	���̩�A3*!

total reward per episode  ��xV�.       ��W�	4��̩�A�* 

Average reward per step۶m�9�kb       ��2	��̩�A�*

epsilon۶m���.       ��W�	_^�̩�A�* 

Average reward per step۶m�ե��       ��2	_�̩�A�*

epsilon۶m�D`�~.       ��W�	W��̩�A�* 

Average reward per step۶m�hK�       ��2	1��̩�A�*

epsilon۶m�R�.       ��W�	� ̩�A�* 

Average reward per step۶m���RR       ��2	�� ̩�A�*

epsilon۶m�
�0.       ��W�	�̩�A�* 

Average reward per step۶m�]g�        ��2	�̩�A�*

epsilon۶m�xz�a.       ��W�	]�̩�A�* 

Average reward per step۶m�&V��       ��2	��̩�A�*

epsilon۶m��C_P.       ��W�	��̩�A�* 

Average reward per step۶m�X��       ��2	��̩�A�*

epsilon۶m�]~b.       ��W�	�h̩�A�* 

Average reward per step۶m��4��       ��2	fi̩�A�*

epsilon۶m� �Bz.       ��W�	�n̩�A�* 

Average reward per step۶m��S#0       ��2	vo̩�A�*

epsilon۶m�G��@.       ��W�	��̩�A�* 

Average reward per step۶m���4       ��2	��̩�A�*

epsilon۶m���.       ��W�	,J̩�A�* 

Average reward per step۶m��L@       ��2	K̩�A�*

epsilon۶m����K.       ��W�	�g̩�A�* 

Average reward per step۶m��(^#       ��2	fh̩�A�*

epsilon۶m����.       ��W�	Y�̩�A�* 

Average reward per step۶m�8b�       ��2	/�̩�A�*

epsilon۶m��?u�.       ��W�	&�̩�A�* 

Average reward per step۶m��v�       ��2	�̩�A�*

epsilon۶m�4�.       ��W�	;̩�A�* 

Average reward per step۶m�®^�       ��2	�;̩�A�*

epsilon۶m�X⎞.       ��W�	l̩�A�* 

Average reward per step۶m�$ �       ��2	�l̩�A�*

epsilon۶m��q�.       ��W�	<�̩�A�* 

Average reward per step۶m���`�       ��2	@�̩�A�*

epsilon۶m����.       ��W�	8�!̩�A�* 

Average reward per step۶m��	}       ��2	��!̩�A�*

epsilon۶m�HhT�.       ��W�	L�#̩�A�* 

Average reward per step۶m�B��Z       ��2	K $̩�A�*

epsilon۶m�	W��.       ��W�	0.&̩�A�* 

Average reward per step۶m�\���       ��2	
/&̩�A�*

epsilon۶m����.       ��W�	-�'̩�A�* 

Average reward per step۶m�$�b       ��2	��'̩�A�*

epsilon۶m��eH%.       ��W�	��)̩�A�* 

Average reward per step۶m�f�       ��2	f�)̩�A�*

epsilon۶m��tϘ0       ���_	I*̩�A4*#
!
Average reward per episode�E���֜�.       ��W�	�*̩�A4*!

total reward per episode  �b[�.       ��W�	��.̩�A�* 

Average reward per step�E���h��       ��2	��.̩�A�*

epsilon�E��2��}.       ��W�	�K0̩�A�* 

Average reward per step�E���OIW       ��2	sL0̩�A�*

epsilon�E�����L.       ��W�	�w2̩�A�* 

Average reward per step�E������       ��2	�x2̩�A�*

epsilon�E�����[.       ��W�	˼4̩�A�* 

Average reward per step�E�����       ��2	��4̩�A�*

epsilon�E���QM.       ��W�	|_6̩�A�* 

Average reward per step�E��+ul�       ��2	$`6̩�A�*

epsilon�E���P��.       ��W�	�8̩�A�* 

Average reward per step�E����*       ��2	̲8̩�A�*

epsilon�E��F�m.       ��W�	��:̩�A�* 

Average reward per step�E����<�       ��2	��:̩�A�*

epsilon�E��U�o�.       ��W�	O�<̩�A�* 

Average reward per step�E����n       ��2	p�<̩�A�*

epsilon�E��Q,�.       ��W�	��>̩�A�* 

Average reward per step�E����       ��2	��>̩�A�*

epsilon�E�����.       ��W�	��@̩�A�* 

Average reward per step�E���oG       ��2	��@̩�A�*

epsilon�E��?��.       ��W�	�C̩�A�* 

Average reward per step�E�����       ��2	��C̩�A�*

epsilon�E���t�.       ��W�	&SE̩�A�* 

Average reward per step�E��06��       ��2	*TE̩�A�*

epsilon�E��*y�.       ��W�	��G̩�A�* 

Average reward per step�E��	�8       ��2	v�G̩�A�*

epsilon�E��LJk&.       ��W�	�-I̩�A�* 

Average reward per step�E��v��       ��2	f.I̩�A�*

epsilon�E����~�.       ��W�	�L̩�A�* 

Average reward per step�E����:�       ��2	�L̩�A�*

epsilon�E���3j�.       ��W�	��M̩�A�* 

Average reward per step�E���!��       ��2	��M̩�A�*

epsilon�E���^.       ��W�	8�O̩�A�* 

Average reward per step�E��w�       ��2	�O̩�A�*

epsilon�E���ư.       ��W�	�Q̩�A�* 

Average reward per step�E��Y�,<       ��2	��Q̩�A�*

epsilon�E����y8.       ��W�	^/T̩�A�* 

Average reward per step�E��"�D,       ��2	@0T̩�A�*

epsilon�E�����.       ��W�	�TV̩�A�* 

Average reward per step�E���r��       ��2	�UV̩�A�*

epsilon�E��=���.       ��W�	��W̩�A�* 

Average reward per step�E�����       ��2	Z�W̩�A�*

epsilon�E�����
.       ��W�	��Y̩�A�* 

Average reward per step�E��b��x       ��2	{�Y̩�A�*

epsilon�E��EVd.       ��W�	0\̩�A�* 

Average reward per step�E��9��o       ��2	�0\̩�A�*

epsilon�E�����.       ��W�	�$^̩�A�* 

Average reward per step�E��B�{e       ��2	�%^̩�A�*

epsilon�E������.       ��W�	�7`̩�A�* 

Average reward per step�E���(�?       ��2	�8`̩�A�*

epsilon�E��E�yI.       ��W�	�b̩�A�* 

Average reward per step�E����l+       ��2	��b̩�A�*

epsilon�E���_��.       ��W�	�d̩�A�* 

Average reward per step�E��q��       ��2	�d̩�A�*

epsilon�E��dV@.       ��W�	)yf̩�A�* 

Average reward per step�E���O��       ��2	�yf̩�A�*

epsilon�E��fv�.       ��W�	��h̩�A�* 

Average reward per step�E�����+       ��2	��h̩�A�*

epsilon�E��5H��.       ��W�	��j̩�A�* 

Average reward per step�E��{*��       ��2	��j̩�A�*

epsilon�E��ցtB.       ��W�	.�m̩�A�* 

Average reward per step�E�����       ��2	�m̩�A�*

epsilon�E���6}�.       ��W�	�q̩�A�* 

Average reward per step�E��1Թ       ��2	Pq̩�A�*

epsilon�E����).       ��W�	|�r̩�A�* 

Average reward per step�E��οy       ��2	��r̩�A�*

epsilon�E��wx�.       ��W�	p&u̩�A�* 

Average reward per step�E���{�       ��2	t'u̩�A�*

epsilon�E���qH30       ���_	��u̩�A5*#
!
Average reward per episode  `�׵[(.       ��W�	J�u̩�A5*!

total reward per episode  �� $.       ��W�	5�y̩�A�* 

Average reward per step  `����       ��2	�y̩�A�*

epsilon  `�&�.       ��W�	Q-{̩�A�* 

Average reward per step  `�����       ��2	.{̩�A�*

epsilon  `�&�V�.       ��W�	�5}̩�A�* 

Average reward per step  `�E�̟       ��2	�<}̩�A�*

epsilon  `� �Ƕ.       ��W�	�j̩�A�* 

Average reward per step  `�א#�       ��2	Yk̩�A�*

epsilon  `�&���.       ��W�	���̩�A�* 

Average reward per step  `��G8       ��2	X��̩�A�*

epsilon  `�pe��.       ��W�	�X�̩�A�* 

Average reward per step  `����[       ��2	�Y�̩�A�*

epsilon  `��~��.       ��W�	�X�̩�A�* 

Average reward per step  `�r	9�       ��2	�Y�̩�A�*

epsilon  `��{�.       ��W�	�̩�A�* 

Average reward per step  `�T�vn       ��2	X�̩�A�*

epsilon  `���<.       ��W�	�:�̩�A�* 

Average reward per step  `��6       ��2	q;�̩�A�*

epsilon  `��WF.       ��W�	M�̩�A�* 

Average reward per step  `���^N       ��2	�M�̩�A�*

epsilon  `��5�.       ��W�	���̩�A�* 

Average reward per step  `��B��       ��2	X��̩�A�*

epsilon  `��.       ��W�	v��̩�A�* 

Average reward per step  `�XT�       ��2	*��̩�A�*

epsilon  `�g��.       ��W�	���̩�A�* 

Average reward per step  `��AW'       ��2	F��̩�A�*

epsilon  `���c�.       ��W�	=(�̩�A�* 

Average reward per step  `�2�K       ��2	�)�̩�A�*

epsilon  `�T�L.       ��W�	���̩�A�* 

Average reward per step  `�Fc�       ��2	Y��̩�A�*

epsilon  `��$��.       ��W�	�H�̩�A�* 

Average reward per step  `��7       ��2	VI�̩�A�*

epsilon  `� �3�.       ��W�	J��̩�A�* 

Average reward per step  `�HYf|       ��2	��̩�A�*

epsilon  `���w.       ��W�	��̩�A�* 

Average reward per step  `��2�x       ��2	U�̩�A�*

epsilon  `�8�	.       ��W�	���̩�A�* 

Average reward per step  `�-VSD       ��2	R��̩�A�*

epsilon  `���b.       ��W�	���̩�A�* 

Average reward per step  `����       ��2	_��̩�A�*

epsilon  `��MA.       ��W�	�6�̩�A�* 

Average reward per step  `��V�B       ��2	�7�̩�A�*

epsilon  `�b�]�.       ��W�	�R�̩�A�* 

Average reward per step  `�ݭ�,       ��2	�S�̩�A�*

epsilon  `��`B�.       ��W�	ƭ̩�A�* 

Average reward per step  `��lo       ��2	�ƭ̩�A�*

epsilon  `���5�.       ��W�	it�̩�A�* 

Average reward per step  `�az�       ��2	`u�̩�A�*

epsilon  `����d.       ��W�	-��̩�A�* 

Average reward per step  `���Q�       ��2	Փ�̩�A�*

epsilon  `�'�Z.       ��W�	���̩�A�* 

Average reward per step  `�-?B)       ��2	=��̩�A�*

epsilon  `���P.       ��W�	4��̩�A�* 

Average reward per step  `�m\�        ��2	���̩�A�*

epsilon  `�sZa.       ��W�	ۦ�̩�A�* 

Average reward per step  `�����       ��2	���̩�A�*

epsilon  `���ѫ.       ��W�	_Խ̩�A�* 

Average reward per step  `���w�       ��2	�Խ̩�A�*

epsilon  `�..y�.       ��W�	~�̩�A�* 

Average reward per step  `�����       ��2	i�̩�A�*

epsilon  `��4.       ��W�	(��̩�A�* 

Average reward per step  `��        ��2	���̩�A�*

epsilon  `�e��.       ��W�	��̩�A�* 

Average reward per step  `���       ��2	���̩�A�*

epsilon  `��D�e.       ��W�	���̩�A�* 

Average reward per step  `��h��       ��2	ѐ�̩�A�*

epsilon  `���-�.       ��W�	l$�̩�A�* 

Average reward per step  `���B       ��2	>%�̩�A�*

epsilon  `��A�.       ��W�	N��̩�A�* 

Average reward per step  `��=m       ��2	���̩�A�*

epsilon  `���S.       ��W�	���̩�A�* 

Average reward per step  `��Y-       ��2	���̩�A�*

epsilon  `�>^�.       ��W�	:��̩�A�* 

Average reward per step  `���$�       ��2	O��̩�A�*

epsilon  `�Ø��.       ��W�	8h�̩�A�* 

Average reward per step  `���       ��2	Qi�̩�A�*

epsilon  `���#(.       ��W�	��̩�A�* 

Average reward per step  `�v W       ��2	ۊ�̩�A�*

epsilon  `��oE.       ��W�	}"�̩�A�* 

Average reward per step  `���?V       ��2	y#�̩�A�*

epsilon  `���I.       ��W�	F@�̩�A�* 

Average reward per step  `��63�       ��2	A�̩�A�*

epsilon  `�-.).       ��W�	k�̩�A�* 

Average reward per step  `�1̫        ��2	�k�̩�A�*

epsilon  `��Tј.       ��W�	���̩�A�* 

Average reward per step  `��w�       ��2	��̩�A�*

epsilon  `��B.       ��W�	��̩�A�* 

Average reward per step  `��� 9       ��2	O�̩�A�*

epsilon  `�H9>�.       ��W�	�C�̩�A�* 

Average reward per step  `��B       ��2	�D�̩�A�*

epsilon  `�+N�.       ��W�	3n�̩�A�* 

Average reward per step  `�|8PS       ��2	 o�̩�A�*

epsilon  `� }��.       ��W�	���̩�A�* 

Average reward per step  `�5�|       ��2	���̩�A�*

epsilon  `�Ȳq�.       ��W�	l^�̩�A�* 

Average reward per step  `�+�Q       ��2	_�̩�A�*

epsilon  `�l%�.       ��W�	�"�̩�A�* 

Average reward per step  `��ÑR       ��2	[#�̩�A�*

epsilon  `�Zk{.       ��W�	���̩�A�* 

Average reward per step  `�1 �S       ��2	{��̩�A�*

epsilon  `�ޏT�.       ��W�	���̩�A�* 

Average reward per step  `�5�X�       ��2	���̩�A�*

epsilon  `����.       ��W�	��̩�A�* 

Average reward per step  `���!*       ��2	��̩�A�*

epsilon  `���E{.       ��W�	T5�̩�A�* 

Average reward per step  `�NV��       ��2	/6�̩�A�*

epsilon  `��=� .       ��W�	���̩�A�* 

Average reward per step  `�XC9j       ��2	���̩�A�*

epsilon  `�/�Ԇ.       ��W�	��̩�A�* 

Average reward per step  `���U       ��2	��̩�A�*

epsilon  `���.       ��W�	�p�̩�A�* 

Average reward per step  `�B�;       ��2	eq�̩�A�*

epsilon  `��A��.       ��W�	��̩�A�* 

Average reward per step  `�aI��       ��2	��̩�A�*

epsilon  `�5�C�.       ��W�	ji̩�A�* 

Average reward per step  `�j��=       ��2	Mj̩�A�*

epsilon  `����,0       ���_	?�̩�A6*#
!
Average reward per episode  ��^�C�.       ��W�	*�̩�A6*!

total reward per episode  ���A.       ��W�	�̩�A�* 

Average reward per step  ����v�       ��2	�̩�A�*

epsilon  �����.       ��W�	)�	̩�A�* 

Average reward per step  ����yM       ��2	��	̩�A�*

epsilon  ��K��.       ��W�	V̩�A�* 

Average reward per step  ��(ư8       ��2	�V̩�A�*

epsilon  �����a.       ��W�	g�̩�A�* 

Average reward per step  ������       ��2	�̩�A�*

epsilon  �����.       ��W�	��̩�A�* 

Average reward per step  ��Wz�6       ��2	L�̩�A�*

epsilon  ��k`E.       ��W�	�G̩�A�* 

Average reward per step  ��u�.       ��2	�H̩�A�*

epsilon  ����,.       ��W�	�k̩�A�* 

Average reward per step  ���F       ��2	�l̩�A�*

epsilon  ��09��.       ��W�	�̩�A�* 

Average reward per step  ��: Ts       ��2	ˢ̩�A�*

epsilon  ��.pA.       ��W�	�;̩�A�* 

Average reward per step  ��O,��       ��2	�<̩�A�*

epsilon  ��3Ʊ\.       ��W�	v�̩�A�* 

Average reward per step  ��Lsr       ��2	e�̩�A�*

epsilon  ��IH��.       ��W�	�V̩�A�* 

Average reward per step  ��а�       ��2	~W̩�A�*

epsilon  ��Qix..       ��W�	U"̩�A�* 

Average reward per step  ���/>�       ��2	"̩�A�*

epsilon  ��У�n.       ��W�	7q$̩�A�* 

Average reward per step  �����R       ��2	mr$̩�A�*

epsilon  ����.       ��W�	��%̩�A�* 

Average reward per step  ��>�i       ��2	Y�%̩�A�*

epsilon  ��r��.       ��W�	
-(̩�A�* 

Average reward per step  ���0�       ��2	�-(̩�A�*

epsilon  ��bi~.       ��W�	��)̩�A�* 

Average reward per step  ����k       ��2	��)̩�A�*

epsilon  ����D.       ��W�	w,̩�A�* 

Average reward per step  ��۽j       ��2	/,̩�A�*

epsilon  ��Q�_.       ��W�	�n.̩�A�* 

Average reward per step  ���]'       ��2	Xo.̩�A�*

epsilon  ��Rc�F.       ��W�	�0̩�A�* 

Average reward per step  ��XH�c       ��2	�0̩�A�*

epsilon  ���`��.       ��W�	�q2̩�A�* 

Average reward per step  ��$l�       ��2	�r2̩�A�*

epsilon  �����.       ��W�	��4̩�A�* 

Average reward per step  ���.{�       ��2	��4̩�A�*

epsilon  ��j䉮.       ��W�	��6̩�A�* 

Average reward per step  �����       ��2	��6̩�A�*

epsilon  ����x.       ��W�	�98̩�A�* 

Average reward per step  ���VG�       ��2	y:8̩�A�*

epsilon  ��ES)�.       ��W�	J�:̩�A�* 

Average reward per step  ��Kn��       ��2	�:̩�A�*

epsilon  ���k�.       ��W�	��<̩�A�* 

Average reward per step  ���٣�       ��2	��<̩�A�*

epsilon  �����.       ��W�	�]A̩�A�* 

Average reward per step  ��M(       ��2	�^A̩�A�*

epsilon  ����g�.       ��W�	��D̩�A�* 

Average reward per step  ��m��       ��2	Q�D̩�A�*

epsilon  ����.       ��W�	LqF̩�A�* 

Average reward per step  ��%���       ��2	6rF̩�A�*

epsilon  �����_.       ��W�	�H̩�A�* 

Average reward per step  ��p�%_       ��2	��H̩�A�*

epsilon  ��?&I�.       ��W�	��J̩�A�* 

Average reward per step  ��k�M�       ��2	0�J̩�A�*

epsilon  ��d�9�.       ��W�	i�L̩�A�* 

Average reward per step  ���-��       ��2	D�L̩�A�*

epsilon  ���Fu&.       ��W�	4LO̩�A�* 

Average reward per step  ���u҉       ��2	'MO̩�A�*

epsilon  �����.       ��W�	��P̩�A�* 

Average reward per step  ���pyx       ��2	��P̩�A�*

epsilon  ��G�f=.       ��W�	.�S̩�A�* 

Average reward per step  ��32�       ��2	.�S̩�A�*

epsilon  �����x.       ��W�	BZW̩�A�* 

Average reward per step  ��x�,�       ��2	�ZW̩�A�*

epsilon  ����0       ���_	%uW̩�A7*#
!
Average reward per episode|�W���/�.       ��W�	�uW̩�A7*!

total reward per episode  ��<�SX.       ��W�	�,]̩�A�* 

Average reward per step|�W��ؚ       ��2	�-]̩�A�*

epsilon|�W��Ǔ�.       ��W�	<�_̩�A�* 

Average reward per step|�W��Ln�       ��2	�_̩�A�*

epsilon|�W�ҟ�k.       ��W�	,+a̩�A�* 

Average reward per step|�W���5       ��2	V,a̩�A�*

epsilon|�W�+X.       ��W�	�{c̩�A�* 

Average reward per step|�W����       ��2	�|c̩�A�*

epsilon|�W����H.       ��W�	��e̩�A�* 

Average reward per step|�W����       ��2	9�e̩�A�*

epsilon|�W��6�.       ��W�	��g̩�A�* 

Average reward per step|�W��>��       ��2	`�g̩�A�*

epsilon|�W�V��.       ��W�	mi̩�A�* 

Average reward per step|�W�N# k       ��2	�mi̩�A�*

epsilon|�W���.       ��W�	�k̩�A�* 

Average reward per step|�W���       ��2	��k̩�A�*

epsilon|�W�f��(.       ��W�	�n̩�A�* 

Average reward per step|�W���.       ��2	Kn̩�A�*

epsilon|�W�$��.       ��W�	Cp̩�A�* 

Average reward per step|�W�>[��       ��2	�Cp̩�A�*

epsilon|�W�!�I�.       ��W�	4�q̩�A�* 

Average reward per step|�W���l       ��2	�q̩�A�*

epsilon|�W���Ğ.       ��W�	�t̩�A�* 

Average reward per step|�W�+]       ��2	�t̩�A�*

epsilon|�W�z&�.       ��W�	X v̩�A�* 

Average reward per step|�W�	sc�       ��2	!!v̩�A�*

epsilon|�W�>�i6.       ��W�	�1x̩�A�* 

Average reward per step|�W��Z�j       ��2	�2x̩�A�*

epsilon|�W��>��.       ��W�	�Uz̩�A�* 

Average reward per step|�W��       ��2	2Wz̩�A�*

epsilon|�W�Eo.       ��W�	kG~̩�A�* 

Average reward per step|�W���=       ��2	RH~̩�A�*

epsilon|�W�'�.       ��W�	t�̩�A�* 

Average reward per step|�W���e       ��2	:�̩�A�*

epsilon|�W�޳�M.       ��W�	c�̩�A�* 

Average reward per step|�W��"@b       ��2	F�̩�A�*

epsilon|�W��j%�.       ��W�	�#�̩�A�* 

Average reward per step|�W��
�S       ��2	�$�̩�A�*

epsilon|�W��GB.       ��W�	T��̩�A�* 

Average reward per step|�W�i�       ��2	\�̩�A�*

epsilon|�W����&.       ��W�	|��̩�A�* 

Average reward per step|�W�5��       ��2	 ��̩�A�*

epsilon|�W����.       ��W�	5�̩�A�* 

Average reward per step|�W���n2       ��2		6�̩�A�*

epsilon|�W�͇�.       ��W�	6#�̩�A�* 

Average reward per step|�W��jO       ��2	$�̩�A�*

epsilon|�W���n.       ��W�	$B�̩�A�* 

Average reward per step|�W��� �       ��2	C�̩�A�*

epsilon|�W�]�D.       ��W�	o֚̩�A�* 

Average reward per step|�W���U?       ��2	Mך̩�A�*

epsilon|�W�� � .       ��W�	�q�̩�A�* 

Average reward per step|�W���\E       ��2	ir�̩�A�*

epsilon|�W��g�.       ��W�	�e�̩�A�* 

Average reward per step|�W���ً       ��2	�f�̩�A�*

epsilon|�W�`5�.       ��W�	�V�̩�A�* 

Average reward per step|�W���$�       ��2	�W�̩�A�*

epsilon|�W����{.       ��W�	�B�̩�A�* 

Average reward per step|�W�� y�       ��2	�C�̩�A�*

epsilon|�W��+h.       ��W�	^�̩�A�* 

Average reward per step|�W�`4�       ��2	^�̩�A�*

epsilon|�W��g�!.       ��W�	px�̩�A�* 

Average reward per step|�W�D��       ��2	�y�̩�A�*

epsilon|�W��	M.       ��W�	b��̩�A�* 

Average reward per step|�W���       ��2	��̩�A�*

epsilon|�W���P�.       ��W�	�y�̩�A�* 

Average reward per step|�W�I�        ��2	�z�̩�A�*

epsilon|�W���A�.       ��W�	Mد̩�A�* 

Average reward per step|�W���{B       ��2	bٯ̩�A�*

epsilon|�W���p�.       ��W�	Xų̩�A�* 

Average reward per step|�W��Q��       ��2	Ƴ̩�A�*

epsilon|�W�M`��.       ��W�	�޵̩�A�* 

Average reward per step|�W�z��L       ��2	�̩�A�*

epsilon|�W�Ԡ�.       ��W�	�*�̩�A�* 

Average reward per step|�W����       ��2	,�̩�A�*

epsilon|�W��:.       ��W�	9չ̩�A�* 

Average reward per step|�W�����       ��2	�չ̩�A�*

epsilon|�W�EOc�.       ��W�	`��̩�A�* 

Average reward per step|�W�~���       ��2	���̩�A�*

epsilon|�W�S���.       ��W�	���̩�A�* 

Average reward per step|�W���       ��2	h��̩�A�*

epsilon|�W�c�!0       ���_	:��̩�A8*#
!
Average reward per episode33;��7�.       ��W�	���̩�A8*!

total reward per episode  �?]�.       ��W�	��̩�A�* 

Average reward per step33;���;�       ��2	��̩�A�*

epsilon33;��@��.       ��W�	X�̩�A�* 

Average reward per step33;���g�       ��2	S�̩�A�*

epsilon33;���.       ��W�	�l�̩�A�* 

Average reward per step33;����       ��2	Dn�̩�A�*

epsilon33;��D�g.       ��W�	�K�̩�A�* 

Average reward per step33;�a��%       ��2	�L�̩�A�*

epsilon33;�AS�.       ��W�	e��̩�A�* 

Average reward per step33;�? <`       ��2	?��̩�A�*

epsilon33;���6�.       ��W�	O�̩�A�* 

Average reward per step33;���       ��2	�O�̩�A�*

epsilon33;�X+t.       ��W�	���̩�A�* 

Average reward per step33;���"N       ��2	���̩�A�*

epsilon33;�!c�.       ��W�	.��̩�A�* 

Average reward per step33;����{       ��2	 ��̩�A�*

epsilon33;����.       ��W�	]6�̩�A�* 

Average reward per step33;�4p       ��2	37�̩�A�*

epsilon33;�d�f4.       ��W�	��̩�A�* 

Average reward per step33;����       ��2	���̩�A�*

epsilon33;��-.       ��W�	�̩�A�* 

Average reward per step33;���       ��2	*�̩�A�*

epsilon33;�>���.       ��W�	��̩�A�* 

Average reward per step33;��
       ��2	��̩�A�*

epsilon33;����.       ��W�	���̩�A�* 

Average reward per step33;�4�d:       ��2	X��̩�A�*

epsilon33;����Q.       ��W�	:�̩�A�* 

Average reward per step33;���E       ��2	�̩�A�*

epsilon33;����.       ��W�	M��̩�A�* 

Average reward per step33;����       ��2	^��̩�A�*

epsilon33;�>��.       ��W�	l�̩�A�* 

Average reward per step33;�&9k       ��2	�l�̩�A�*

epsilon33;����I.       ��W�	#��̩�A�* 

Average reward per step33;��49j       ��2	���̩�A�*

epsilon33;����s.       ��W�	)$�̩�A�* 

Average reward per step33;���K       ��2	�$�̩�A�*

epsilon33;�
U��.       ��W�	�F�̩�A�* 

Average reward per step33;��       ��2	�G�̩�A�*

epsilon33;��>��.       ��W�	�h�̩�A�* 

Average reward per step33;�g-       ��2	�i�̩�A�*

epsilon33;��,kC.       ��W�	��̩�A�* 

Average reward per step33;���F       ��2	���̩�A�*

epsilon33;��wذ.       ��W�	��̩�A�* 

Average reward per step33;�r�P�       ��2	��̩�A�*

epsilon33;��<�.       ��W�	��̩�A�* 

Average reward per step33;��{��       ��2	���̩�A�*

epsilon33;�Tw�.       ��W�	=��̩�A�* 

Average reward per step33;���       ��2	؝�̩�A�*

epsilon33;��=��.       ��W�	4��̩�A�* 

Average reward per step33;�F���       ��2	��̩�A�*

epsilon33;����~.       ��W�	�6 ̩�A�* 

Average reward per step33;�ʞΆ       ��2	�7 ̩�A�*

epsilon33;�}��.       ��W�	N� ̩�A�* 

Average reward per step33;�q��       ��2	�� ̩�A�*

epsilon33;��L�.       ��W�	
� ̩�A�* 

Average reward per step33;�K.p�       ��2	� ̩�A�*

epsilon33;��kx.       ��W�	(G ̩�A�* 

Average reward per step33;�?��       ��2	�G ̩�A�*

epsilon33;�@��0       ���_	<f ̩�A9*#
!
Average reward per episode�������{.       ��W�	�f ̩�A9*!

total reward per episode  Õc�	.       ��W�	�F ̩�A�* 

Average reward per step����4P�       ��2	ZG ̩�A�*

epsilon�����jd%.       ��W�	dv ̩�A�* 

Average reward per step�������       ��2	2w ̩�A�*

epsilon�����P�.       ��W�	� ̩�A�* 

Average reward per step�������       ��2	z� ̩�A�*

epsilon����\铤.       ��W�	h� ̩�A�* 

Average reward per step��������       ��2	B� ̩�A�*

epsilon����u@��.       ��W�	�G ̩�A�* 

Average reward per step�����l��       ��2	�H ̩�A�*

epsilon�����%�`.       ��W�	�� ̩�A�* 

Average reward per step�����v��       ��2	H� ̩�A�*

epsilon����5߭�.       ��W�	c� ̩�A�* 

Average reward per step����H� g       ��2	E� ̩�A�*

epsilon����9n��.       ��W�	� ̩�A�* 

Average reward per step����qj�-       ��2	* ̩�A�*

epsilon�����6�C.       ��W�	��  ̩�A�* 

Average reward per step����?��       ��2	��  ̩�A�*

epsilon���� �[.       ��W�	# ̩�A�* 

Average reward per step��������       ��2	�# ̩�A�*

epsilon����^X,.       ��W�	�b% ̩�A�* 

Average reward per step����R�P�       ��2	�c% ̩�A�*

epsilon������9?.       ��W�	��' ̩�A�* 

Average reward per step��������       ��2	��' ̩�A�*

epsilon����h���.       ��W�	�-) ̩�A�* 

Average reward per step�����HK�       ��2	j.) ̩�A�*

epsilon������t�.       ��W�	]+ ̩�A�* 

Average reward per step����={�       ��2	�]+ ̩�A�*

epsilon�����-��.       ��W�	&�- ̩�A�* 

Average reward per step�����;�       ��2	��- ̩�A�*

epsilon����y˚�.       ��W�	;S0 ̩�A�* 

Average reward per step������U       ��2	T0 ̩�A�*

epsilon������"O.       ��W�	��3 ̩�A�* 

Average reward per step�����4       ��2	��3 ̩�A�*

epsilon�����}*.       ��W�	��5 ̩�A�* 

Average reward per step������8       ��2	��5 ̩�A�*

epsilon����6\��0       ���_	6 ̩�A:*#
!
Average reward per episode  �V`4.       ��W�	�6 ̩�A:*!

total reward per episode  +�2X�(.       ��W�	�P: ̩�A�* 

Average reward per step  �>j2       ��2	zQ: ̩�A�*

epsilon  �L	��.       ��W�	O> ̩�A�* 

Average reward per step  �[/�8       ��2	�> ̩�A�*

epsilon  �F`̣.       ��W�	�?@ ̩�A�* 

Average reward per step  �3g�       ��2	�@@ ̩�A�*

epsilon  �qNY�.       ��W�	oB ̩�A�* 

Average reward per step  �yԼu       ��2	�oB ̩�A�*

epsilon  �Yq`:.       ��W�	WD ̩�A�* 

Average reward per step  �l�.�       ��2	%D ̩�A�*

epsilon  �l��.       ��W�	��F ̩�A�* 

Average reward per step  ���p       ��2	q�F ̩�A�*

epsilon  ��a/.       ��W�	��J ̩�A�* 

Average reward per step  �l@�       ��2	��J ̩�A�*

epsilon  ���p.       ��W�	3L ̩�A�* 

Average reward per step  ��d�       ��2	�3L ̩�A�*

epsilon  ��*�.       ��W�	�lN ̩�A�* 

Average reward per step  �n̝       ��2	�nN ̩�A�*

epsilon  ���d�.       ��W�	�Q ̩�A�* 

Average reward per step  ��]�U       ��2	�Q ̩�A�*

epsilon  ����.       ��W�	ǄT ̩�A�* 

Average reward per step  �'�4F       ��2	U�T ̩�A�*

epsilon  �&Jo6.       ��W�	h�V ̩�A�* 

Average reward per step  �����       ��2	J�V ̩�A�*

epsilon  �Q�u�.       ��W�	��X ̩�A�* 

Average reward per step  �����       ��2	A�X ̩�A�*

epsilon  �ŏi.       ��W�	�[ ̩�A�* 

Average reward per step  �C�h�       ��2	|[ ̩�A�*

epsilon  �4�C.       ��W�	)]] ̩�A�* 

Average reward per step  �m�j�       ��2	�]] ̩�A�*

epsilon  �461.       ��W�	��^ ̩�A�* 

Average reward per step  ���       ��2	��^ ̩�A�*

epsilon  ��"T.       ��W�	%a ̩�A�* 

Average reward per step  ���       ��2	�%a ̩�A�*

epsilon  �ȯ*.       ��W�	�yc ̩�A�* 

Average reward per step  ��ȃ@       ��2	�zc ̩�A�*

epsilon  �N�|�.       ��W�	^�d ̩�A�* 

Average reward per step  ��3Y       ��2	�d ̩�A�*

epsilon  ���.z.       ��W�	^hg ̩�A�* 

Average reward per step  ����
       ��2	<ig ̩�A�*

epsilon  �ϫKg.       ��W�	�rj ̩�A�* 

Average reward per step  �"}y�       ��2	�sj ̩�A�*

epsilon  �}$�+0       ���_	��j ̩�A;*#
!
Average reward per episode=���-��.       ��W�	!�j ̩�A;*!

total reward per episode   ßV��.       ��W�	l�o ̩�A�* 

Average reward per step=���^�X�       ��2	�o ̩�A�*

epsilon=����aӀ.       ��W�	��q ̩�A�* 

Average reward per step=���.|�       ��2	a�q ̩�A�*

epsilon=����r�.       ��W�	�os ̩�A�* 

Average reward per step=���0"-)       ��2	qs ̩�A�*

epsilon=����,�.       ��W�	G�u ̩�A�* 

Average reward per step=�������       ��2	�u ̩�A�*

epsilon=����=�.       ��W�	�w ̩�A�* 

Average reward per step=���.�2*       ��2	��w ̩�A�*

epsilon=����,��.       ��W�	��y ̩�A�* 

Average reward per step=����1"�       ��2	��y ̩�A�*

epsilon=���G��A.       ��W�	[D| ̩�A�* 

Average reward per step=���[�<       ��2	E| ̩�A�*

epsilon=���f׆s.       ��W�	��} ̩�A�* 

Average reward per step=���8���       ��2	��} ̩�A�*

epsilon=���8�I.       ��W�	�2� ̩�A�* 

Average reward per step=���+��       ��2	�3� ̩�A�*

epsilon=���2@�.       ��W�	wځ ̩�A�* 

Average reward per step=���b��}       ��2	Iہ ̩�A�*

epsilon=����
z.       ��W�	l� ̩�A�* 

Average reward per step=���L��Y       ��2	F� ̩�A�*

epsilon=����mg.       ��W�	X˅ ̩�A�* 

Average reward per step=���{"	       ��2	�̅ ̩�A�*

epsilon=���ݔr.       ��W�	�� ̩�A�* 

Average reward per step=���Y�K�       ��2	k� ̩�A�*

epsilon=�����3�.       ��W�	�
� ̩�A�* 

Average reward per step=���^k�:       ��2	�� ̩�A�*

epsilon=������.       ��W�	Jy� ̩�A�* 

Average reward per step=������       ��2	:z� ̩�A�*

epsilon=����P�}.       ��W�	\� ̩�A�* 

Average reward per step=���9�o       ��2	�� ̩�A�*

epsilon=���t5q�.       ��W�	�O� ̩�A�* 

Average reward per step=���:       ��2	PP� ̩�A�*

epsilon=���X��i.       ��W�	��� ̩�A�* 

Average reward per step=����1�0       ��2	|�� ̩�A�*

epsilon=���0�o�.       ��W�	�֔ ̩�A�* 

Average reward per step=���e��[       ��2	�ה ̩�A�*

epsilon=���0o�>.       ��W�	I�� ̩�A�* 

Average reward per step=���c^W�       ��2	퀘 ̩�A�*

epsilon=�����s.       ��W�	ܡ� ̩�A�* 

Average reward per step=����z       ��2	Ƣ� ̩�A�*

epsilon=���StY�.       ��W�	+ ̩�A�* 

Average reward per step=���)0�       ��2	� ̩�A�*

epsilon=����(с.       ��W�	��� ̩�A�* 

Average reward per step=������       ��2	^�� ̩�A�*

epsilon=����rW�.       ��W�	=�� ̩�A�* 

Average reward per step=����!T�       ��2	�� ̩�A�*

epsilon=�����.       ��W�	hТ ̩�A�* 

Average reward per step=���yg�       ��2	(Ѣ ̩�A�*

epsilon=���bo��.       ��W�	� � ̩�A�* 

Average reward per step=����ڬ9       ��2	�� ̩�A�*

epsilon=���/��m.       ��W�	Z�� ̩�A�* 

Average reward per step=���Ϻ��       ��2	I�� ̩�A�*

epsilon=�������.       ��W�	6� ̩�A�* 

Average reward per step=���o��<       ��2	�6� ̩�A�*

epsilon=�����@.       ��W�	��� ̩�A�* 

Average reward per step=���S�[x       ��2	R�� ̩�A�*

epsilon=���B�<.       ��W�	׬ ̩�A�* 

Average reward per step=���c��       ��2	�׬ ̩�A�*

epsilon=�������.       ��W�	N� ̩�A�* 

Average reward per step=���`	�)       ��2	�N� ̩�A�*

epsilon=�����u0       ���_	�n� ̩�A<*#
!
Average reward per episode����VY�.       ��W�	mo� ̩�A<*!

total reward per episode  ���0Q.       ��W�	_A� ̩�A�* 

Average reward per step������       ��2	B� ̩�A�*

epsilon�����W�.       ��W�	�k� ̩�A�* 

Average reward per step����K��       ��2	�l� ̩�A�*

epsilon������*.       ��W�	�� ̩�A�* 

Average reward per step����'��w       ��2	��� ̩�A�*

epsilon�����е�.       ��W�	�_� ̩�A�* 

Average reward per step��������       ��2	N`� ̩�A�*

epsilon����!U�&.       ��W�	Rպ ̩�A�* 

Average reward per step����(O|1       ��2	�պ ̩�A�*

epsilon������Q.       ��W�	�+� ̩�A�* 

Average reward per step������i9       ��2	�,� ̩�A�*

epsilon����b���.       ��W�	/�� ̩�A�* 

Average reward per step�����TY�       ��2	߈� ̩�A�*

epsilon����+Dk�.       ��W�	�� ̩�A�* 

Average reward per step������z       ��2	1�� ̩�A�*

epsilon����fR��.       ��W�	�s� ̩�A�* 

Average reward per step������Ag       ��2	�t� ̩�A�*

epsilon�����+�.       ��W�	ظ� ̩�A�* 

Average reward per step�������       ��2	��� ̩�A�*

epsilon����o2e.       ��W�	7 � ̩�A�* 

Average reward per step�����Ԓ       ��2	u� ̩�A�*

epsilon�����#�.       ��W�	��� ̩�A�* 

Average reward per step����_6       ��2	��� ̩�A�*

epsilon������!k.       ��W�	0�� ̩�A�* 

Average reward per step�����?�x       ��2	��� ̩�A�*

epsilon�����̡D.       ��W�	w�� ̩�A�* 

Average reward per step������       ��2	I�� ̩�A�*

epsilon�����S�.       ��W�	h� ̩�A�* 

Average reward per step����e��       ��2	)� ̩�A�*

epsilon����^�Զ.       ��W�	-!� ̩�A�* 

Average reward per step����!9\n       ��2	�!� ̩�A�*

epsilon������F�.       ��W�	,�� ̩�A�* 

Average reward per step�����.��       ��2	؁� ̩�A�*

epsilon�����!�;.       ��W�	�� ̩�A�* 

Average reward per step����@�g       ��2	C� ̩�A�*

epsilon�����C.       ��W�	b� ̩�A�* 

Average reward per step����G�R       ��2	c� ̩�A�*

epsilon����gX3f.       ��W�	��� ̩�A�* 

Average reward per step����m�Fm       ��2	Ț� ̩�A�*

epsilon��������.       ��W�	��� ̩�A�* 

Average reward per step����?]��       ��2	��� ̩�A�*

epsilon����x-�h.       ��W�	�D� ̩�A�* 

Average reward per step�����1٥       ��2	oE� ̩�A�*

epsilon����\���.       ��W�	�b� ̩�A�* 

Average reward per step������y       ��2	cc� ̩�A�*

epsilon������D�.       ��W�	�� ̩�A�* 

Average reward per step�����Cr       ��2	��� ̩�A�*

epsilon����D�}\.       ��W�	�� ̩�A�* 

Average reward per step������my       ��2	��� ̩�A�*

epsilon�����3�.       ��W�	�v� ̩�A�* 

Average reward per step��������       ��2	�w� ̩�A�*

epsilon����F�5].       ��W�	հ� ̩�A�* 

Average reward per step�������       ��2	��� ̩�A�*

epsilon������.       ��W�	� � ̩�A�* 

Average reward per step�����Wp�       ��2	�� ̩�A�*

epsilon����~��4.       ��W�	�&� ̩�A�* 

Average reward per step�����Q�       ��2	�'� ̩�A�*

epsilon�������*.       ��W�	7� ̩�A�* 

Average reward per step����*B��       ��2	�7� ̩�A�*

epsilon�����]�.       ��W�	U�� ̩�A�* 

Average reward per step�����竞       ��2	�� ̩�A�*

epsilon����bUQ3.       ��W�	8.� ̩�A�* 

Average reward per step����r�$�       ��2	�.� ̩�A�*

epsilon����).0.       ��W�	_�� ̩�A�* 

Average reward per step������`       ��2	p�� ̩�A�*

epsilon����dY�.       ��W�	�� ̩�A�* 

Average reward per step������+�       ��2	�� ̩�A�*

epsilon�����lS.       ��W�	Eh!̩�A�* 

Average reward per step����Ѩ��       ��2	8i!̩�A�*

epsilon������1.       ��W�	��!̩�A�* 

Average reward per step�����-�       ��2	��!̩�A�*

epsilon����d;�].       ��W�	�\!̩�A�* 

Average reward per step����,���       ��2	-]!̩�A�*

epsilon����HӫQ.       ��W�	��!̩�A�* 

Average reward per step����5�z�       ��2	�!̩�A�*

epsilon����m@�q.       ��W�	��	!̩�A�* 

Average reward per step������o�       ��2	x�	!̩�A�*

epsilon�����4�.       ��W�	8,!̩�A�* 

Average reward per step����	!r       ��2	<.!̩�A�*

epsilon������.       ��W�	��!̩�A�* 

Average reward per step����z��$       ��2		�!̩�A�*

epsilon����`��9.       ��W�		8!̩�A�* 

Average reward per step�����;�F       ��2	9!̩�A�*

epsilon����5Lr.       ��W�	��!̩�A�* 

Average reward per step����̚b8       ��2	6�!̩�A�*

epsilon������E�.       ��W�	�W!̩�A�* 

Average reward per step�������       ��2	�X!̩�A�*

epsilon������ي.       ��W�	��!̩�A�* 

Average reward per step����}��       ��2	��!̩�A�*

epsilon����c��.       ��W�	9B!̩�A�* 

Average reward per step������A�       ��2	�D!̩�A�*

epsilon����a��.0       ���_	$d!̩�A=*#
!
Average reward per episode   ���u.       ��W�	�d!̩�A=*!

total reward per episode  ��p�)�.       ��W�	T�!̩�A�* 

Average reward per step   �r���       ��2	�!̩�A�*

epsilon   �od�.       ��W�	E. !̩�A�* 

Average reward per step   ����J       ��2	</ !̩�A�*

epsilon   ��7�4.       ��W�	.9#!̩�A�* 

Average reward per step   �#�       ��2	:#!̩�A�*

epsilon   �:�\.       ��W�	)&!̩�A�* 

Average reward per step   �V]�       ��2	E*&!̩�A�*

epsilon   �O�.       ��W�	)!̩�A�* 

Average reward per step   ��j�       ��2	�)!̩�A�*

epsilon   ��Y;.       ��W�	�u+!̩�A�* 

Average reward per step   ���uc       ��2	�v+!̩�A�*

epsilon   ��1��.       ��W�	Af.!̩�A�* 

Average reward per step   �L�n       ��2	#g.!̩�A�*

epsilon   �-��.       ��W�		�0!̩�A�* 

Average reward per step   �~��       ��2	��0!̩�A�*

epsilon   �
?#.       ��W�	X:3!̩�A�* 

Average reward per step   �X�T�       ��2	�:3!̩�A�*

epsilon   �\�G.       ��W�	��4!̩�A�* 

Average reward per step   �$��T       ��2	��4!̩�A�*

epsilon   �� &P.       ��W�	;7!̩�A�* 

Average reward per step   ���Z�       ��2	*7!̩�A�*

epsilon   ���B.       ��W�	�k9!̩�A�* 

Average reward per step   ��9'�       ��2	/l9!̩�A�*

epsilon   ���K.       ��W�	l�;!̩�A�* 

Average reward per step   ��A�       ��2	ĕ;!̩�A�*

epsilon   �{�A.       ��W�	,=!̩�A�* 

Average reward per step   ����W       ��2	�=!̩�A�*

epsilon   ���.
.       ��W�	Ec?!̩�A�* 

Average reward per step   �%�S       ��2	�d?!̩�A�*

epsilon   ��q�F.       ��W�	zA!̩�A�* 

Average reward per step   �����       ��2	iA!̩�A�*

epsilon   �]B\#.       ��W�	��C!̩�A�* 

Average reward per step   ����       ��2	(�C!̩�A�*

epsilon   ����0       ���_	��C!̩�A>*#
!
Average reward per episode����3j�.       ��W�	D�C!̩�A>*!

total reward per episode  Âއ�.       ��W�	�H!̩�A�* 

Average reward per step����8]c       ��2	[H!̩�A�*

epsilon����gx�.       ��W�	��K!̩�A�* 

Average reward per step�����m       ��2	��K!̩�A�*

epsilon���k�/f.       ��W�	dN!̩�A�* 

Average reward per step���p��[       ��2	N!̩�A�*

epsilon����@�(.       ��W�	h�O!̩�A�* 

Average reward per step���3q-G       ��2	B�O!̩�A�*

epsilon���m��.       ��W�		�Q!̩�A�* 

Average reward per step����&[�       ��2	��Q!̩�A�*

epsilon�����L[.       ��W�	1CT!̩�A�* 

Average reward per step�������       ��2	�CT!̩�A�*

epsilon����*��.       ��W�	?�V!̩�A�* 

Average reward per step�����4�       ��2	�V!̩�A�*

epsilon���8��E.       ��W�	��Z!̩�A�* 

Average reward per step���5&�       ��2	L�Z!̩�A�*

epsilon����ʤ�.       ��W�	<�^!̩�A�* 

Average reward per step������       ��2	
�^!̩�A�*

epsilon�����O.       ��W�	��_!̩�A�* 

Average reward per step����Xۍ       ��2	~�_!̩�A�*

epsilon���B��'.       ��W�	w�b!̩�A�* 

Average reward per step�����.�       ��2	^�b!̩�A�*

epsilon���ca�.       ��W�	�vd!̩�A�* 

Average reward per step���E�       ��2	�wd!̩�A�*

epsilon���q�Dw.       ��W�	��f!̩�A�* 

Average reward per step����e�       ��2	t�f!̩�A�*

epsilon������.       ��W�	v�h!̩�A�* 

Average reward per step����F�m       ��2	]�h!̩�A�*

epsilon������.       ��W�	c�j!̩�A�* 

Average reward per step���}���       ��2	R�j!̩�A�*

epsilon����v<�.       ��W�	�Qm!̩�A�* 

Average reward per step���r���       ��2	�Rm!̩�A�*

epsilon�����/p.       ��W�	s�o!̩�A�* 

Average reward per step���7H�       ��2	N�o!̩�A�*

epsilon����C0	.       ��W�	�zq!̩�A�* 

Average reward per step���A���       ��2	�{q!̩�A�*

epsilon���ӱ��.       ��W�	��s!̩�A�* 

Average reward per step���P,�       ��2	��s!̩�A�*

epsilon�����ǹ.       ��W�	�tv!̩�A�* 

Average reward per step����)a�       ��2	�uv!̩�A�*

epsilon�����j3.       ��W�	2�y!̩�A�* 

Average reward per step����ݶ�       ��2	��y!̩�A�*

epsilon���#{.       ��W�	�{!̩�A�* 

Average reward per step�����?�       ��2	��{!̩�A�*

epsilon���?��.       ��W�	FC~!̩�A�* 

Average reward per step������       ��2	�C~!̩�A�*

epsilon����<W?.       ��W�	��!̩�A�* 

Average reward per step���P�E�       ��2	s�!̩�A�*

epsilon�����T.       ��W�	�5�!̩�A�* 

Average reward per step����wI       ��2	�6�!̩�A�*

epsilon���p	?.       ��W�	�l�!̩�A�* 

Average reward per step����g�       ��2	am�!̩�A�*

epsilon����Y_�.       ��W�	d��!̩�A�* 

Average reward per step���";�@       ��2	G��!̩�A�*

epsilon�����P.       ��W�	Z�!̩�A�* 

Average reward per step���m{]�       ��2	��!̩�A�*

epsilon����v�.       ��W�	���!̩�A�* 

Average reward per step���б�       ��2	v��!̩�A�*

epsilon���v�3�.       ��W�	�\�!̩�A�* 

Average reward per step���nTV�       ��2	�]�!̩�A�*

epsilon������.       ��W�	y�!̩�A�* 

Average reward per step���r��2       ��2	G�!̩�A�*

epsilon���gTO.       ��W�	�u�!̩�A�* 

Average reward per step���<��(       ��2	yv�!̩�A�*

epsilon�����7.       ��W�	Rd�!̩�A�* 

Average reward per step����65       ��2	Ie�!̩�A�*

epsilon�������.       ��W�	���!̩�A�* 

Average reward per step���Q�̭       ��2	n��!̩�A�*

epsilon���q5��0       ���_	m�!̩�A?*#
!
Average reward per episodexxX���	i.       ��W�	:�!̩�A?*!

total reward per episode  ���B'�.       ��W�	V՚!̩�A�* 

Average reward per stepxxX�P-5�       ��2	�՚!̩�A�*

epsilonxxX��+��.       ��W�	R��!̩�A�* 

Average reward per stepxxX����&       ��2	I��!̩�A�*

epsilonxxX��4X.       ��W�	�Ȟ!̩�A�* 

Average reward per stepxxX�p���       ��2	qɞ!̩�A�*

epsilonxxX����e.       ��W�	k�!̩�A�* 

Average reward per stepxxX���]�       ��2	0�!̩�A�*

epsilonxxX���0�.       ��W�	8��!̩�A�* 

Average reward per stepxxX�W��#       ��2	��!̩�A�*

epsilonxxX����>.       ��W�	��!̩�A�* 

Average reward per stepxxX���4	       ��2	l�!̩�A�*

epsilonxxX�LD�l.       ��W�	"q�!̩�A�* 

Average reward per stepxxX�9;[]       ��2	�q�!̩�A�*

epsilonxxX�2Ƶ.       ��W�	W|�!̩�A�* 

Average reward per stepxxX�o e       ��2	9}�!̩�A�*

epsilonxxX���E.       ��W�	ǽ�!̩�A�* 

Average reward per stepxxX�`<S�       ��2	���!̩�A�*

epsilonxxX�ϊt.       ��W�	?�!̩�A�* 

Average reward per stepxxX�97       ��2	�!̩�A�*

epsilonxxX��s��.       ��W�	�w�!̩�A�* 

Average reward per stepxxX�0�n       ��2	px�!̩�A�*

epsilonxxX��G�.       ��W�	#��!̩�A�* 

Average reward per stepxxX���       ��2	麳!̩�A�*

epsilonxxX�,��.       ��W�	���!̩�A�* 

Average reward per stepxxX�L�c�       ��2	���!̩�A�*

epsilonxxX�E[.       ��W�	&�!̩�A�* 

Average reward per stepxxX�[#��       ��2	�&�!̩�A�*

epsilonxxX�!�r.       ��W�	PĹ!̩�A�* 

Average reward per stepxxX�^4`�       ��2	7Ź!̩�A�*

epsilonxxX�W���.       ��W�	��!̩�A�* 

Average reward per stepxxX�b�d       ��2	#�!̩�A�*

epsilonxxX�)��t.       ��W�	�I�!̩�A�* 

Average reward per stepxxX����       ��2	MJ�!̩�A�*

epsilonxxX�_��L.       ��W�	�!̩�A�* 

Average reward per stepxxX�����       ��2	
�!̩�A�*

epsilonxxX�-��0       ���_	�O�!̩�A@*#
!
Average reward per episode�	����.       ��W�	<P�!̩�A@*!

total reward per episode  �1��$.       ��W�	`��!̩�A�* 

Average reward per step�	�̢!�       ��2	��!̩�A�*

epsilon�	�" �.       ��W�	� �!̩�A�* 

Average reward per step�	�i��c       ��2	�!�!̩�A�*

epsilon�	�U��.       ��W�	��!̩�A�* 

Average reward per step�	�e{��       ��2	L��!̩�A�*

epsilon�	����.       ��W�	1�!̩�A�* 

Average reward per step�	��a��       ��2	�!̩�A�*

epsilon�	��蝵.       ��W�	���!̩�A�* 

Average reward per step�	�6��
       ��2	���!̩�A�*

epsilon�	�O�}�.       ��W�	D��!̩�A�* 

Average reward per step�	��o�       ��2	���!̩�A�*

epsilon�	�uiz_.       ��W�	��!̩�A�* 

Average reward per step�	�DY��       ��2	y�!̩�A�*

epsilon�	�EO!.       ��W�	�:�!̩�A�* 

Average reward per step�	��ˎ�       ��2	d;�!̩�A�*

epsilon�	�|��=.       ��W�	;��!̩�A�* 

Average reward per step�	��!s�       ��2	��!̩�A�*

epsilon�	�!���.       ��W�	;4�!̩�A�* 

Average reward per step�	�x�+       ��2	i5�!̩�A�*

epsilon�	���N.       ��W�	���!̩�A�* 

Average reward per step�	���j�       ��2	���!̩�A�*

epsilon�	�r��{.       ��W�	[&�!̩�A�* 

Average reward per step�	�B��#       ��2	x'�!̩�A�*

epsilon�	�G�-[.       ��W�	W��!̩�A�* 

Average reward per step�	���5       ��2	W��!̩�A�*

epsilon�	��;�;.       ��W�	�8�!̩�A�* 

Average reward per step�	��¨�       ��2	�:�!̩�A�*

epsilon�	����.       ��W�	M�!̩�A�* 

Average reward per step�	���       ��2	�!̩�A�*

epsilon�	���B.       ��W�	S{�!̩�A�* 

Average reward per step�	�f)��       ��2	c|�!̩�A�*

epsilon�	��M7.       ��W�	}�!̩�A�* 

Average reward per step�	�=�ݦ       ��2	t�!̩�A�*

epsilon�	�s(V].       ��W�	�*�!̩�A�* 

Average reward per step�	�T}�       ��2	,�!̩�A�*

epsilon�	�`y�.       ��W�	���!̩�A�* 

Average reward per step�	�Wd[�       ��2	b��!̩�A�*

epsilon�	���.       ��W�	��!̩�A�* 

Average reward per step�	���S       ��2	��!̩�A�*

epsilon�	�$�R�.       ��W�	��!̩�A�* 

Average reward per step�	�d�ڑ       ��2	b�!̩�A�*

epsilon�	��q��.       ��W�	��!̩�A�* 

Average reward per step�	���\       ��2	���!̩�A�*

epsilon�	�"�s.       ��W�	���!̩�A�* 

Average reward per step�	�u��A       ��2	��!̩�A�*

epsilon�	����.       ��W�	���!̩�A�* 

Average reward per step�	�m�d       ��2	���!̩�A�*

epsilon�	�y�6�.       ��W�	��"̩�A�* 

Average reward per step�	�e;j�       ��2	��"̩�A�*

epsilon�	��ڄ�.       ��W�	m"̩�A�* 

Average reward per step�	�@���       ��2	n"̩�A�*

epsilon�	�8qƢ.       ��W�		"̩�A�* 

Average reward per step�	�ً��       ��2	�	"̩�A�*

epsilon�	�W{�@.       ��W�	��"̩�A�* 

Average reward per step�	��~4       ��2	��"̩�A�*

epsilon�	�X�.       ��W�	��"̩�A�* 

Average reward per step�	��SYQ       ��2	��"̩�A�*

epsilon�	�0X�.       ��W�	�"̩�A�* 

Average reward per step�	��9?       ��2	�"̩�A�*

epsilon�	�P�4�.       ��W�	��"̩�A�* 

Average reward per step�	��DS       ��2	��"̩�A�*

epsilon�	��׻0       ���_	�"̩�AA*#
!
Average reward per episode�֚�a�Y.       ��W�	M"̩�AA*!

total reward per episode  ø%i.       ��W�	��"̩�A�* 

Average reward per step�֚��t�       ��2	��"̩�A�*

epsilon�֚�6�CI.       ��W�	Y3!"̩�A�* 

Average reward per step�֚��!t�       ��2	z4!"̩�A�*

epsilon�֚��d,<.       ��W�	��#"̩�A�* 

Average reward per step�֚��r��       ��2	�#"̩�A�*

epsilon�֚�|�i�.       ��W�	�	("̩�A�* 

Average reward per step�֚�ٔ�X       ��2	�
("̩�A�*

epsilon�֚�gj�.       ��W�	+l+"̩�A�* 

Average reward per step�֚�t�tR       ��2	am+"̩�A�*

epsilon�֚��e��.       ��W�	jl."̩�A�* 

Average reward per step�֚�Ӧf       ��2	�m."̩�A�*

epsilon�֚��ux.       ��W�	�2"̩�A�* 

Average reward per step�֚��y�       ��2	!2"̩�A�*

epsilon�֚���.       ��W�	|�4"̩�A�* 

Average reward per step�֚�|�       ��2	��4"̩�A�*

epsilon�֚��a�o.       ��W�	?�8"̩�A�* 

Average reward per step�֚�Z�X(       ��2	q�8"̩�A�*

epsilon�֚���3.       ��W�	�\<"̩�A�* 

Average reward per step�֚��r��       ��2	�]<"̩�A�*

epsilon�֚�.�[.       ��W�	J�>"̩�A�* 

Average reward per step�֚��Wv       ��2	x�>"̩�A�*

epsilon�֚���B.       ��W�	m�@"̩�A�* 

Average reward per step�֚��q,�       ��2	��@"̩�A�*

epsilon�֚�����.       ��W�	�B"̩�A�* 

Average reward per step�֚��-�       ��2	0�B"̩�A�*

epsilon�֚��?`�.       ��W�	7�E"̩�A�* 

Average reward per step�֚��Ӫ       ��2	��E"̩�A�*

epsilon�֚���+@.       ��W�		�I"̩�A�* 

Average reward per step�֚�ے��       ��2		�I"̩�A�*

epsilon�֚�cm�.       ��W�	�@M"̩�A�* 

Average reward per step�֚��.N�       ��2	�AM"̩�A�*

epsilon�֚�"��C.       ��W�	�O"̩�A�* 

Average reward per step�֚���:       ��2	b�O"̩�A�*

epsilon�֚��|.       ��W�	bhS"̩�A�* 

Average reward per step�֚��>U�       ��2	�iS"̩�A�*

epsilon�֚��N.       ��W�	��U"̩�A�* 

Average reward per step�֚��\��       ��2	��U"̩�A�*

epsilon�֚�J<c.       ��W�	
Z"̩�A�* 

Average reward per step�֚�"HR�       ��2	�
Z"̩�A�*

epsilon�֚�}��.       ��W�	��["̩�A�* 

Average reward per step�֚��)�       ��2	n�["̩�A�*

epsilon�֚����.       ��W�	;r^"̩�A�* 

Average reward per step�֚���o�       ��2	�r^"̩�A�*

epsilon�֚�2���.       ��W�	.:b"̩�A�* 

Average reward per step�֚��ӸM       ��2	6;b"̩�A�*

epsilon�֚�� 8a0       ���_	�nb"̩�AB*#
!
Average reward per episode����9�K.       ��W�	�ob"̩�AB*!

total reward per episode  �R�a.       ��W�	��h"̩�A�* 

Average reward per step�����)z       ��2	��h"̩�A�*

epsilon�����:�.       ��W�	$�l"̩�A�* 

Average reward per step�������m       ��2	g�l"̩�A�*

epsilon����\�<�.       ��W�	Q�p"̩�A�* 

Average reward per step����Fc@�       ��2	j�p"̩�A�*

epsilon����"�.       ��W�	�.u"̩�A�* 

Average reward per step������b�       ��2	�/u"̩�A�*

epsilon�����h��.       ��W�	�y"̩�A�* 

Average reward per step����� [       ��2	�y"̩�A�*

epsilon����S�:�.       ��W�	�|"̩�A�* 

Average reward per step����sO��       ��2	�|"̩�A�*

epsilon������
.       ��W�	i�"̩�A�* 

Average reward per step����6�-]       ��2	ʩ"̩�A�*

epsilon�����~D?.       ��W�	���"̩�A�* 

Average reward per step����j4�s       ��2	��"̩�A�*

epsilon�����H��.       ��W�	��"̩�A�* 

Average reward per step����r2.       ��2	"̩�A�*

epsilon����x�,E.       ��W�	���"̩�A�* 

Average reward per step����+c�o       ��2	���"̩�A�*

epsilon������t.       ��W�	�5�"̩�A�* 

Average reward per step����DF�       ��2	�6�"̩�A�*

epsilon����.�5.       ��W�	FЍ"̩�A�* 

Average reward per step������o#       ��2	�э"̩�A�*

epsilon����Sj�7.       ��W�	cБ"̩�A�* 

Average reward per step�����*3�       ��2	ґ"̩�A�*

epsilon����O)�.       ��W�	}>�"̩�A�* 

Average reward per step����E�г       ��2	�?�"̩�A�*

epsilon����Bb��.       ��W�	x��"̩�A�* 

Average reward per step������B=       ��2	g��"̩�A�*

epsilon�����(��.       ��W�	��"̩�A�* 

Average reward per step����D�       ��2	���"̩�A�*

epsilon�����\�v.       ��W�	�Q�"̩�A�* 

Average reward per step����ɷO�       ��2	vR�"̩�A�*

epsilon����'���.       ��W�	�P�"̩�A�* 

Average reward per step����Pc\�       ��2	�Q�"̩�A�*

epsilon����b�{�.       ��W�	��"̩�A�* 

Average reward per step������j       ��2	��"̩�A�*

epsilon�����wZ4.       ��W�	į�"̩�A�* 

Average reward per step�����	%:       ��2	���"̩�A�*

epsilon�������&.       ��W�	R��"̩�A�* 

Average reward per step�����[�t       ��2	'��"̩�A�*

epsilon������.       ��W�	��"̩�A�* 

Average reward per step����:���       ��2	��"̩�A�*

epsilon����[�u.       ��W�	�Ҷ"̩�A�* 

Average reward per step�����u�       ��2	�ն"̩�A�*

epsilon���� ��.       ��W�	Tq�"̩�A�* 

Average reward per step����u�[       ��2	~r�"̩�A�*

epsilon�����x&�.       ��W�	��"̩�A�* 

Average reward per step�����a       ��2	4�"̩�A�*

epsilon�������.       ��W�	�ݽ"̩�A�* 

Average reward per step����#�-Y       ��2	�޽"̩�A�*

epsilon������$N.       ��W�	q��"̩�A�* 

Average reward per step������       ��2	ѯ�"̩�A�*

epsilon����*��.       ��W�	%�"̩�A�* 

Average reward per step����"h=�       ��2	&�"̩�A�*

epsilon����m��=0       ���_	�i�"̩�AC*#
!
Average reward per episode  ���ǫ\.       ��W�	�j�"̩�AC*!

total reward per episode  ��Y�.       ��W�	~��"̩�A�* 

Average reward per step  ������       ��2	O��"̩�A�*

epsilon  ��aks.       ��W�	��"̩�A�* 

Average reward per step  �����[       ��2	��"̩�A�*

epsilon  ���v.       ��W�	� �"̩�A�* 

Average reward per step  ����#�       ��2	2�"̩�A�*

epsilon  ��`��.       ��W�	n��"̩�A�* 

Average reward per step  ����^r       ��2	j��"̩�A�*

epsilon  ���Q�G.       ��W�	7��"̩�A�* 

Average reward per step  ����<"       ��2	��"̩�A�*

epsilon  ���Y�W.       ��W�	���"̩�A�* 

Average reward per step  ���ϴ�       ��2	���"̩�A�*

epsilon  ��^��.       ��W�	O��"̩�A�* 

Average reward per step  ���mo�       ��2	��"̩�A�*

epsilon  ���4�.       ��W�	���"̩�A�* 

Average reward per step  �����       ��2	���"̩�A�*

epsilon  ��]���.       ��W�	�N�"̩�A�* 

Average reward per step  ������       ��2	�O�"̩�A�*

epsilon  ����&.       ��W�	2��"̩�A�* 

Average reward per step  �� �`"       ��2	B��"̩�A�*

epsilon  ���@�~.       ��W�	+n�"̩�A�* 

Average reward per step  ���YR�       ��2	o�"̩�A�*

epsilon  ��sϯ�.       ��W�	�8�"̩�A�* 

Average reward per step  ����{-       ��2	�9�"̩�A�*

epsilon  ���y��.       ��W�	���"̩�A�* 

Average reward per step  ��G�       ��2	���"̩�A�*

epsilon  ��}SD�.       ��W�	��"̩�A�* 

Average reward per step  ����u       ��2	]��"̩�A�*

epsilon  ���p�H.       ��W�	p\#̩�A�* 

Average reward per step  ��e��]       ��2	�]#̩�A�*

epsilon  ���;�.       ��W�	C�#̩�A�* 

Average reward per step  ����D(       ��2	��#̩�A�*

epsilon  ��v�.       ��W�	L�#̩�A�* 

Average reward per step  ����-       ��2	��#̩�A�*

epsilon  ��!�W�.       ��W�	k#̩�A�* 

Average reward per step  ����       ��2	Ql#̩�A�*

epsilon  ��,��.       ��W�	��#̩�A�* 

Average reward per step  ��[�G�       ��2	ö#̩�A�*

epsilon  ���q�.       ��W�	H�#̩�A�* 

Average reward per step  �����       ��2	.�#̩�A�*

epsilon  ��!��.       ��W�	s�#̩�A�* 

Average reward per step  �����       ��2	ܟ#̩�A�*

epsilon  ����8�.       ��W�	H#̩�A�* 

Average reward per step  ��?���       ��2	�#̩�A�*

epsilon  ���*�.       ��W�	s�#̩�A�* 

Average reward per step  ����?        ��2	��#̩�A�*

epsilon  ��7���.       ��W�	�K#̩�A�* 

Average reward per step  ���{;       ��2	�L#̩�A�*

epsilon  ��Ai�!.       ��W�	�`"#̩�A�* 

Average reward per step  ���0{�       ��2	�a"#̩�A�*

epsilon  ���M.       ��W�	��$#̩�A�* 

Average reward per step  ����4�       ��2	��$#̩�A�*

epsilon  ��l��.       ��W�	-x(#̩�A�* 

Average reward per step  �����       ��2	-y(#̩�A�*

epsilon  ��߼�0       ���_	ܞ(#̩�AD*#
!
Average reward per episode�K��L|�y.       ��W�	ϟ(#̩�AD*!

total reward per episode  ��#|.       ��W�	Q�.#̩�A�* 

Average reward per step�K���֍       ��2	/�.#̩�A�*

epsilon�K���	4i.       ��W�	ڬ2#̩�A�* 

Average reward per step�K��O�K�       ��2	��2#̩�A�*

epsilon�K��� �.       ��W�	��6#̩�A�* 

Average reward per step�K�����       ��2	<�6#̩�A�*

epsilon�K���ɝ+.       ��W�	t�8#̩�A�* 

Average reward per step�K����       ��2	��8#̩�A�*

epsilon�K��y�bY.       ��W�	4�<#̩�A�* 

Average reward per step�K����Q#       ��2		�<#̩�A�*

epsilon�K��;3��.       ��W�	�g?#̩�A�* 

Average reward per step�K����       ��2	�h?#̩�A�*

epsilon�K��B��<.       ��W�	eS�#̩�A�* 

Average reward per step�K��Г�       ��2	CT�#̩�A�*

epsilon�K��=��.       ��W�	)̇#̩�A�* 

Average reward per step�K������       ��2	>͇#̩�A�*

epsilon�K��~�.       ��W�	��#̩�A�* 

Average reward per step�K��٧�       ��2	���#̩�A�*

epsilon�K���z��.       ��W�	Ѱ�#̩�A�* 

Average reward per step�K���G��       ��2	��#̩�A�*

epsilon�K��Ƚ.       ��W�	ᕓ#̩�A�* 

Average reward per step�K����^       ��2	��#̩�A�*

epsilon�K�����.       ��W�	�>�#̩�A�* 

Average reward per step�K��ΰu!       ��2	@�#̩�A�*

epsilon�K��-���.       ��W�	eޙ#̩�A�* 

Average reward per step�K���')       ��2	���#̩�A�*

epsilon�K��(�,�.       ��W�	�#̩�A�* 

Average reward per step�K��I&�        ��2	6��#̩�A�*

epsilon�K��7�/�.       ��W�	��#̩�A�* 

Average reward per step�K���9��       ��2	���#̩�A�*

epsilon�K��	��.       ��W�	DP�#̩�A�* 

Average reward per step�K��F�O�       ��2	DQ�#̩�A�*

epsilon�K���#�`.       ��W�	��#̩�A�* 

Average reward per step�K��P�R�       ��2	�	�#̩�A�*

epsilon�K��0�.       ��W�	�ۥ#̩�A�* 

Average reward per step�K���{-]       ��2	ݥ#̩�A�*

epsilon�K����K�.       ��W�	jk�#̩�A�* 

Average reward per step�K���->U       ��2	�l�#̩�A�*

epsilon�K���B].       ��W�	��#̩�A�* 

Average reward per step�K��4�&       ��2	��#̩�A�*

epsilon�K���ֶ.       ��W�	�~�#̩�A�* 

Average reward per step�K�����       ��2	��#̩�A�*

epsilon�K����Y�.       ��W�	Qذ#̩�A�* 

Average reward per step�K���t;�       ��2	bٰ#̩�A�*

epsilon�K��gx^�.       ��W�	��#̩�A�* 

Average reward per step�K��`f@       ��2	�#̩�A�*

epsilon�K���#M�0       ���_	��#̩�AE*#
!
Average reward per episodez���Or��.       ��W�	��#̩�AE*!

total reward per episode  ���v�.       ��W�	s�#̩�A�* 

Average reward per stepz���z��e       ��2	s�#̩�A�*

epsilonz����3��.       ��W�	��#̩�A�* 

Average reward per stepz����f�C       ��2	��#̩�A�*

epsilonz�������.       ��W�	��#̩�A�* 

Average reward per stepz����k��       ��2	#̩�A�*

epsilonz�����:F.       ��W�	�v�#̩�A�* 

Average reward per stepz���#5�K       ��2	�w�#̩�A�*

epsilonz�����=.       ��W�	���#̩�A�* 

Average reward per stepz���Ww       ��2	5��#̩�A�*

epsilonz������.       ��W�	ё�#̩�A�* 

Average reward per stepz�������       ��2	)��#̩�A�*

epsilonz�����$G.       ��W�	��#̩�A�* 

Average reward per stepz������`       ��2	ҋ�#̩�A�*

epsilonz���A��.       ��W�	f��#̩�A�* 

Average reward per stepz������       ��2	���#̩�A�*

epsilonz���|+�P.       ��W�	���#̩�A�* 

Average reward per stepz���8z�	       ��2	��#̩�A�*

epsilonz���T�Ÿ.       ��W�	�w�#̩�A�* 

Average reward per stepz�����s�       ��2	�x�#̩�A�*

epsilonz����\>.       ��W�	IK�#̩�A�* 

Average reward per stepz���T�       ��2	L�#̩�A�*

epsilonz���]6�".       ��W�	<�#̩�A�* 

Average reward per stepz����`�)       ��2	]�#̩�A�*

epsilonz�����.       ��W�	UO�#̩�A�* 

Average reward per stepz����(       ��2	/P�#̩�A�*

epsilonz����sBB.       ��W�	$
�#̩�A�* 

Average reward per stepz���XQ]       ��2	9�#̩�A�*

epsilonz���_�@.       ��W�	��#̩�A�* 

Average reward per stepz���K��       ��2	j�#̩�A�*

epsilonz����ś�.       ��W�	�d�#̩�A�* 

Average reward per stepz���>��       ��2	�e�#̩�A�*

epsilonz����Va�.       ��W�	F~�#̩�A�* 

Average reward per stepz�����0�       ��2	�~�#̩�A�*

epsilonz����¨".       ��W�	�R�#̩�A�* 

Average reward per stepz���{���       ��2	eS�#̩�A�*

epsilonz���d[��.       ��W�	���#̩�A�* 

Average reward per stepz���&�~       ��2	s��#̩�A�*

epsilonz���,ό�.       ��W�	5��#̩�A�* 

Average reward per stepz����*%C       ��2	ݲ�#̩�A�*

epsilonz�����P.       ��W�	��#̩�A�* 

Average reward per stepz����u��       ��2	��#̩�A�*

epsilonz������.       ��W�	�F�#̩�A�* 

Average reward per stepz����M-       ��2	kG�#̩�A�*

epsilonz����z.       ��W�	D��#̩�A�* 

Average reward per stepz���~�WF       ��2	z��#̩�A�*

epsilonz����S6�.       ��W�	��#̩�A�* 

Average reward per stepz���|�*�       ��2	H�#̩�A�*

epsilonz�����7.       ��W�	�e�#̩�A�* 

Average reward per stepz����{f�       ��2	�f�#̩�A�*

epsilonz����&�&.       ��W�	���#̩�A�* 

Average reward per stepz�����       ��2	P��#̩�A�*

epsilonz����C �.       ��W�	�X�#̩�A�* 

Average reward per stepz���:�W�       ��2	�Y�#̩�A�*

epsilonz���$��.       ��W�	¦�#̩�A�* 

Average reward per stepz���6}&�       ��2	���#̩�A�*

epsilonz���,�i'.       ��W�	�4�#̩�A�* 

Average reward per stepz�����^�       ��2	�5�#̩�A�*

epsilonz���2��.       ��W�	�i�#̩�A�* 

Average reward per stepz���[�O       ��2	nj�#̩�A�*

epsilonz���֨f�.       ��W�	r��#̩�A�* 

Average reward per stepz���m��P       ��2	��#̩�A�*

epsilonz����1/�.       ��W�	��$̩�A�* 

Average reward per stepz���D�       ��2	w�$̩�A�*

epsilonz������0       ���_	'�$̩�AF*#
!
Average reward per episode  r�����.       ��W�	��$̩�AF*!

total reward per episode  �§Z|�.       ��W�	�	$̩�A�* 

Average reward per step  r����       ��2	��	$̩�A�*

epsilon  r��2(b.       ��W�	K�L$̩�A�* 

Average reward per step  r�=S�       ��2	��L$̩�A�*

epsilon  r�ܗ�.       ��W�	i�O$̩�A�* 

Average reward per step  r��8>�       ��2	K�O$̩�A�*

epsilon  r�+ƈ..       ��W�	�tR$̩�A�* 

Average reward per step  r��A�       ��2	�uR$̩�A�*

epsilon  r�5�5�.       ��W�	{�U$̩�A�* 

Average reward per step  r����       ��2	D�U$̩�A�*

epsilon  r��xAv.       ��W�	1]X$̩�A�* 

Average reward per step  r�0��+       ��2	 ^X$̩�A�*

epsilon  r�k}*�.       ��W�	H�Z$̩�A�* 

Average reward per step  r����N       ��2	�Z$̩�A�*

epsilon  r�EU�.       ��W�	˼^$̩�A�* 

Average reward per step  r�� +6       ��2	f�^$̩�A�*

epsilon  r�,ܻ\.       ��W�	�-b$̩�A�* 

Average reward per step  r�5�       ��2	�.b$̩�A�*

epsilon  r�d��.       ��W�	�xd$̩�A�* 

Average reward per step  r�1���       ��2	>yd$̩�A�*

epsilon  r��\��.       ��W�	G�f$̩�A�* 

Average reward per step  r��\
�       ��2	%�f$̩�A�*

epsilon  r��#��.       ��W�	h$̩�A�* 

Average reward per step  r�t�2       ��2	�h$̩�A�*

epsilon  r�'�^�.       ��W�	G>j$̩�A�* 

Average reward per step  r����/       ��2	�>j$̩�A�*

epsilon  r��g��.       ��W�	B^l$̩�A�* 

Average reward per step  r��&�h       ��2	$_l$̩�A�*

epsilon  r�3pQ.       ��W�	�n$̩�A�* 

Average reward per step  r��x��       ��2	Ȕn$̩�A�*

epsilon  r��_..       ��W�	#�p$̩�A�* 

Average reward per step  r���+�       ��2	�p$̩�A�*

epsilon  r�Rq.       ��W�	
�r$̩�A�* 

Average reward per step  r����       ��2	�r$̩�A�*

epsilon  r��ђ�.       ��W�	��t$̩�A�* 

Average reward per step  r�Q��k       ��2	��t$̩�A�*

epsilon  r�X�K(.       ��W�	a�v$̩�A�* 

Average reward per step  r��`��       ��2	7�v$̩�A�*

epsilon  r��CT�0       ���_	p�v$̩�AG*#
!
Average reward per episodey�7$��.       ��W�	1�v$̩�AG*!

total reward per episode  Å���.       ��W�	l�z$̩�A�* 

Average reward per stepy��s�       ��2	O�z$̩�A�*

epsilony�%f
�.       ��W�	V�|$̩�A�* 

Average reward per stepy�;�       ��2	(�|$̩�A�*

epsilony�˯J5.       ��W�	@$̩�A�* 

Average reward per stepy�G��       ��2	�@$̩�A�*

epsilony�����.       ��W�	��$̩�A�* 

Average reward per stepy���&�       ��2	x�$̩�A�*

epsilony��@.       ��W�	�#�$̩�A�* 

Average reward per stepy�r�!k       ��2	t$�$̩�A�*

epsilony�$nj*.       ��W�	��$̩�A�* 

Average reward per stepy�����       ��2	��$̩�A�*

epsilony�I7p.       ��W�	'/�$̩�A�* 

Average reward per stepy��J��       ��2	
0�$̩�A�*

epsilony��4}�.       ��W�	s��$̩�A�* 

Average reward per stepy��]�       ��2	U��$̩�A�*

epsilony��B�.       ��W�	�i�$̩�A�* 

Average reward per stepy�n��       ��2	�j�$̩�A�*

epsilony��.       ��W�	���$̩�A�* 

Average reward per stepy���ʱ       ��2	ܹ�$̩�A�*

epsilony�J���.       ��W�	A�$̩�A�* 

Average reward per stepy�5��       ��2	��$̩�A�*

epsilony��:��.       ��W�	 ė$̩�A�* 

Average reward per stepy�<s�{       ��2	�ė$̩�A�*

epsilony���U.       ��W�	_$�$̩�A�* 

Average reward per stepy�����       ��2	5%�$̩�A�*

epsilony����.       ��W�	���$̩�A�* 

Average reward per stepy�QMԘ       ��2	��$̩�A�*

epsilony��@A.       ��W�	V՝$̩�A�* 

Average reward per stepy���s       ��2	,֝$̩�A�*

epsilony�E�Ѓ.       ��W�	h�$̩�A�* 

Average reward per stepy��5H�       ��2	: �$̩�A�*

epsilony�1�.       ��W�	��$̩�A�* 

Average reward per stepy���f�       ��2	/�$̩�A�*

epsilony�e���0       ���_	�3�$̩�AH*#
!
Average reward per episode��!�Faw�.       ��W�	v4�$̩�AH*!

total reward per episode  ,á��.       ��W�	X��$̩�A�* 

Average reward per step��!�K��]       ��2	:��$̩�A�*

epsilon��!�S��.       ��W�	b��$̩�A�* 

Average reward per step��!�
&�        ��2	+��$̩�A�*

epsilon��!�֤.       ��W�	|�$̩�A�* 

Average reward per step��!�@GR*       ��2	�|�$̩�A�*

epsilon��!�G�%�.       ��W�	쥴$̩�A�* 

Average reward per step��!��*��       ��2	���$̩�A�*

epsilon��!�S�	P.       ��W�	�j�$̩�A�* 

Average reward per step��!��=~�       ��2	�k�$̩�A�*

epsilon��!��l�H.       ��W�	��$̩�A�* 

Average reward per step��!�a�M       ��2	£�$̩�A�*

epsilon��!���~^.       ��W�	���$̩�A�* 

Average reward per step��!���n       ��2	s��$̩�A�*

epsilon��!��/(.       ��W�	�|�$̩�A�* 

Average reward per step��!��I		       ��2	�}�$̩�A�*

epsilon��!�^���.       ��W�	ݲ�$̩�A�* 

Average reward per step��!�@f|       ��2	���$̩�A�*

epsilon��!�~�~i.       ��W�	L7�$̩�A�* 

Average reward per step��!�o��0       ��2	�7�$̩�A�*

epsilon��!�w�'C.       ��W�	�>�$̩�A�* 

Average reward per step��!���1       ��2	�?�$̩�A�*

epsilon��!��3i.       ��W�	���$̩�A�* 

Average reward per step��!�X�E�       ��2	���$̩�A�*

epsilon��!����.       ��W�	��$̩�A�* 

Average reward per step��!��P�]       ��2	��$̩�A�*

epsilon��!�j���.       ��W�	m��$̩�A�* 

Average reward per step��!�ӕ3}       ��2	H��$̩�A�*

epsilon��!�z�p.       ��W�	h��$̩�A�* 

Average reward per step��!�9t�       ��2	W��$̩�A�*

epsilon��!��҉v.       ��W�	F��$̩�A�* 

Average reward per step��!�M�w       ��2	��$̩�A�*

epsilon��!���[.       ��W�	���$̩�A�* 

Average reward per step��!��IFe       ��2	j��$̩�A�*

epsilon��!���5F.       ��W�	d:�$̩�A�* 

Average reward per step��!��*�       ��2	<�$̩�A�*

epsilon��!�����.       ��W�	#��$̩�A�* 

Average reward per step��!�e\)�       ��2	��$̩�A�*

epsilon��!���ģ.       ��W�	x��$̩�A�* 

Average reward per step��!��#,�       ��2	R��$̩�A�*

epsilon��!�ՖQ�.       ��W�	�\�$̩�A�* 

Average reward per step��!��N}       ��2	t]�$̩�A�*

epsilon��!�Wu�T.       ��W�	j��$̩�A�* 

Average reward per step��!�8��       ��2		��$̩�A�*

epsilon��!�ȵ8�.       ��W�	�.�$̩�A�* 

Average reward per step��!�� F�       ��2	j/�$̩�A�*

epsilon��!�aE�.       ��W�	~�$̩�A�* 

Average reward per step��!�V��:       ��2	�$̩�A�*

epsilon��!���.       ��W�	���$̩�A�* 

Average reward per step��!�҇��       ��2	x��$̩�A�*

epsilon��!�SA0       ���_	H��$̩�AI*#
!
Average reward per episode����~`�.       ��W�	��$̩�AI*!

total reward per episode  Ý�&2.       ��W�	���$̩�A�* 

Average reward per step����f       ��2	���$̩�A�*

epsilon���DY�.       ��W�	χ�$̩�A�* 

Average reward per step���
B��       ��2	v��$̩�A�*

epsilon���׺I.       ��W�	���$̩�A�* 

Average reward per step���nT�6       ��2	X��$̩�A�*

epsilon�����2.       ��W�	���$̩�A�* 

Average reward per step�����       ��2	���$̩�A�*

epsilon���+V��.       ��W�	�I�$̩�A�* 

Average reward per step���U�.�       ��2	�J�$̩�A�*

epsilon����\GK.       ��W�	���$̩�A�* 

Average reward per step����B:v       ��2	���$̩�A�*

epsilon�����"�.       ��W�	r��$̩�A�* 

Average reward per step���Sv�,       ��2	���$̩�A�*

epsilon���º}|.       ��W�	��$̩�A�* 

Average reward per step���@N	       ��2	��$̩�A�*

epsilon���VD'�.       ��W�	Cr�$̩�A�* 

Average reward per step���r�       ��2	&s�$̩�A�*

epsilon���e5�.       ��W�	�M�$̩�A�* 

Average reward per step����Y^       ��2	+N�$̩�A�*

epsilon����c�.       ��W�	E��$̩�A�* 

Average reward per step���j1�i       ��2	���$̩�A�*

epsilon����A�$.       ��W�	W�$̩�A�* 

Average reward per step����\�       ��2	6�$̩�A�*

epsilon�����).       ��W�	I��$̩�A�* 

Average reward per step���}�w       ��2	/��$̩�A�*

epsilon����@�.       ��W�	�� %̩�A�* 

Average reward per step���I��.       ��2	n� %̩�A�*

epsilon���c�"�.       ��W�	f.%̩�A�* 

Average reward per step���e�m�       ��2	E/%̩�A�*

epsilon���]�z .       ��W�	�^%̩�A�* 

Average reward per step�����C�       ��2	�_%̩�A�*

epsilon���hd��.       ��W�	z%̩�A�* 

Average reward per step�����G       ��2	�z%̩�A�*

epsilon���[�r�.       ��W�	�%̩�A�* 

Average reward per step���4�       ��2	��%̩�A�*

epsilon����o|.       ��W�	E�
%̩�A�* 

Average reward per step������       ��2	��
%̩�A�*

epsilon������L.       ��W�	�_%̩�A�* 

Average reward per step�����|�       ��2	�`%̩�A�*

epsilon����XUH.       ��W�	��%̩�A�* 

Average reward per step���w3 �       ��2	x�%̩�A�*

epsilon���1�.       ��W�	\v%̩�A�* 

Average reward per step����2-0       ��2	)w%̩�A�*

epsilon���r�a.       ��W�	_b%̩�A�* 

Average reward per step���# ʴ       ��2	�b%̩�A�*

epsilon������'0       ���_	̵%̩�AJ*#
!
Average reward per episoded!��R�_.       ��W�	��%̩�AJ*!

total reward per episode  �dr��.       ��W�	�f%̩�A�* 

Average reward per stepd!��"�*(       ��2	jg%̩�A�*

epsilond!��[��.       ��W�	��%̩�A�* 

Average reward per stepd!����[       ��2	��%̩�A�*

epsilond!�����.       ��W�	�?%̩�A�* 

Average reward per stepd!���r�w       ��2	�@%̩�A�*

epsilond!��v<.       ��W�	��%̩�A�* 

Average reward per stepd!���]P�       ��2	��%̩�A�*

epsilond!���߬b.       ��W�	�?!%̩�A�* 

Average reward per stepd!���]       ��2	�@!%̩�A�*

epsilond!��+��.       ��W�	Ί#%̩�A�* 

Average reward per stepd!����       ��2	z�#%̩�A�*

epsilond!��c�9.       ��W�	�%%̩�A�* 

Average reward per stepd!��{��       ��2	w%%̩�A�*

epsilond!����.       ��W�	�7'%̩�A�* 

Average reward per stepd!��z���       ��2	�8'%̩�A�*

epsilond!��5���.       ��W�	��)%̩�A�* 

Average reward per stepd!��4��w       ��2	a�)%̩�A�*

epsilond!��@k�r.       ��W�	��+%̩�A�* 

Average reward per stepd!��Z.@       ��2	��+%̩�A�*

epsilond!�����(.       ��W�	�)-%̩�A�* 

Average reward per stepd!��uQ�       ��2	c*-%̩�A�*

epsilond!��*kR�.       ��W�	Xq/%̩�A�* 

Average reward per stepd!���Y.       ��2	Gr/%̩�A�*

epsilond!���#�2.       ��W�	��1%̩�A�* 

Average reward per stepd!���䎧       ��2	��1%̩�A�*

epsilond!���;t.       ��W�	+�3%̩�A�* 

Average reward per stepd!���4J       ��2	ף3%̩�A�*

epsilond!�����y.       ��W�	��5%̩�A�* 

Average reward per stepd!��S!>       ��2	}�5%̩�A�*

epsilond!��z �.       ��W�	AI7%̩�A�* 

Average reward per stepd!����       ��2	J7%̩�A�*

epsilond!��xetJ.       ��W�	�|9%̩�A�* 

Average reward per stepd!��$|��       ��2	5}9%̩�A�*

epsilond!���13.       ��W�	��;%̩�A�* 

Average reward per stepd!��d%Y�       ��2	��;%̩�A�*

epsilond!���	.       ��W�	�>%̩�A�* 

Average reward per stepd!��&�0�       ��2	)>%̩�A�*

epsilond!��\�{�.       ��W�	�:@%̩�A�* 

Average reward per stepd!����%�       ��2	:;@%̩�A�*

epsilond!���A��.       ��W�	��A%̩�A�* 

Average reward per stepd!���5��       ��2	S�A%̩�A�*

epsilond!��ͥ��.       ��W�	D%̩�A�* 

Average reward per stepd!��Z�       ��2	"D%̩�A�*

epsilond!��קǋ.       ��W�	oEF%̩�A�* 

Average reward per stepd!��M=	;       ��2	�FF%̩�A�*

epsilond!��M���.       ��W�	�lH%̩�A�* 

Average reward per stepd!������       ��2	PmH%̩�A�*

epsilond!���'?�.       ��W�	J�J%̩�A�* 

Average reward per stepd!������       ��2	��J%̩�A�*

epsilond!��q�%�.       ��W�	�9L%̩�A�* 

Average reward per stepd!���߯�       ��2	u:L%̩�A�*

epsilond!���l�.       ��W�	/�N%̩�A�* 

Average reward per stepd!���ٖ�       ��2	��N%̩�A�*

epsilond!������0       ���_	q�N%̩�AK*#
!
Average reward per episode�����N�.       ��W�	�N%̩�AK*!

total reward per episode  ÅC��.       ��W�	��T%̩�A�* 

Average reward per step�����>Bd       ��2	��T%̩�A�*

epsilon����4��n.       ��W�	��V%̩�A�* 

Average reward per step�����O5       ��2	8�V%̩�A�*

epsilon����EF��.       ��W�	���%̩�A�* 

Average reward per step����P`�       ��2	n��%̩�A�*

epsilon����^J[.       ��W�	��%̩�A�* 

Average reward per step�����@��       ��2	���%̩�A�*

epsilon����_��e.       ��W�	)�%̩�A�* 

Average reward per step������l       ��2	�)�%̩�A�*

epsilon�����6T_.       ��W�	ʣ�%̩�A�* 

Average reward per step�������6       ��2	���%̩�A�*

epsilon����[(0�.       ��W�	���%̩�A�* 

Average reward per step����y�(�       ��2	���%̩�A�*

epsilon����B��.       ��W�	���%̩�A�* 

Average reward per step����c���       ��2	���%̩�A�*

epsilon������Bo.       ��W�	Me�%̩�A�* 

Average reward per step����X
       ��2	'f�%̩�A�*

epsilon����`YS.       ��W�	<��%̩�A�* 

Average reward per step����u��[       ��2	���%̩�A�*

epsilon����!�:.       ��W�	L6�%̩�A�* 

Average reward per step����	�?}       ��2		7�%̩�A�*

epsilon����_'�c.       ��W�	PU�%̩�A�* 

Average reward per step����j�/�       ��2	�U�%̩�A�*

epsilon����J��*.       ��W�	7o�%̩�A�* 

Average reward per step������M       ��2	p�%̩�A�*

epsilon����B[�.       ��W�	���%̩�A�* 

Average reward per step����ضV,       ��2	{��%̩�A�*

epsilon�����+p.       ��W�	_	�%̩�A�* 

Average reward per step����K<e5       ��2	W
�%̩�A�*

epsilon��������.       ��W�	0��%̩�A�* 

Average reward per step�����ep�       ��2	���%̩�A�*

epsilon����KdV5.       ��W�	�û%̩�A�* 

Average reward per step����饕a       ��2	vĻ%̩�A�*

epsilon������� .       ��W�	l	�%̩�A�* 

Average reward per step����w��       ��2	_
�%̩�A�*

epsilon���� Ք.       ��W�	���%̩�A�* 

Average reward per step����r�4+       ��2	���%̩�A�*

epsilon����p$C.       ��W�	���%̩�A�* 

Average reward per step������c       ��2	M��%̩�A�*

epsilon�����.       ��W�	� �%̩�A�* 

Average reward per step�����&Ȗ       ��2	�!�%̩�A�*

epsilon����!�k�.       ��W�	���%̩�A�* 

Average reward per step����9��]       ��2	N��%̩�A�*

epsilon����'��0.       ��W�	��%̩�A�* 

Average reward per step�����L�       ��2	K�%̩�A�*

epsilon����g�8.       ��W�	vP�%̩�A�* 

Average reward per step������<       ��2	vQ�%̩�A�*

epsilon����y���.       ��W�	���%̩�A�* 

Average reward per step�����w�H       ��2	���%̩�A�*

epsilon����W�D�.       ��W�	FE�%̩�A�* 

Average reward per step������!       ��2	|F�%̩�A�*

epsilon������/�0       ���_	.��%̩�AL*#
!
Average reward per episode�ع�T��.       ��W�	%��%̩�AL*!

total reward per episode  á�y".       ��W�	`�%̩�A�* 

Average reward per step�ع�B|T       ��2	�`�%̩�A�*

epsilon�ع�?�o.       ��W�	��%̩�A�* 

Average reward per step�ع���ò       ��2	��%̩�A�*

epsilon�ع�8���.       ��W�	&��%̩�A�* 

Average reward per step�ع�(�z       ��2	��%̩�A�*

epsilon�ع��6�r.       ��W�	�y�%̩�A�* 

Average reward per step�ع���g�       ��2	Sz�%̩�A�*

epsilon�ع����.       ��W�	���%̩�A�* 

Average reward per step�ع�(,�?       ��2	���%̩�A�*

epsilon�ع��Fb%.       ��W�	�%̩�A�* 

Average reward per step�ع�j]�       ��2	��%̩�A�*

epsilon�ع��G��.       ��W�	�~�%̩�A�* 

Average reward per step�ع��aZe       ��2	x�%̩�A�*

epsilon�ع��}�.       ��W�	Ǹ�%̩�A�* 

Average reward per step�ع�Д�]       ��2	���%̩�A�*

epsilon�ع�MFv�.       ��W�	��%̩�A�* 

Average reward per step�ع�99��       ��2	��%̩�A�*

epsilon�ع����.       ��W�	��%̩�A�* 

Average reward per step�ع��(\�       ��2	ҏ�%̩�A�*

epsilon�ع�ʡS.       ��W�	g�%̩�A�* 

Average reward per step�ع���t       ��2	�%̩�A�*

epsilon�ع�5j�v.       ��W�	��%̩�A�* 

Average reward per step�ع����       ��2	���%̩�A�*

epsilon�ع����.       ��W�	�U�%̩�A�* 

Average reward per step�ع�(\�       ��2	2W�%̩�A�*

epsilon�ع�z���.       ��W�	���%̩�A�* 

Average reward per step�ع���       ��2	���%̩�A�*

epsilon�ع��&�.       ��W�	�R�%̩�A�* 

Average reward per step�ع��)�       ��2	�S�%̩�A�*

epsilon�ع�~Mrm.       ��W�	�s�%̩�A�* 

Average reward per step�ع����I       ��2	�t�%̩�A�*

epsilon�ع�Rs��.       ��W�	���%̩�A�* 

Average reward per step�ع�DFW�       ��2	k��%̩�A�*

epsilon�ع�2N��.       ��W�	���%̩�A�* 

Average reward per step�ع���K       ��2	���%̩�A�*

epsilon�ع��ܴ�.       ��W�	I��%̩�A�* 

Average reward per step�ع�o�y�       ��2	��%̩�A�*

epsilon�ع��d.       ��W�	 &̩�A�* 

Average reward per step�ع��%��       ��2	� &̩�A�*

epsilon�ع����C.       ��W�	"�&̩�A�* 

Average reward per step�ع��c4
       ��2	Ί&̩�A�*

epsilon�ع�&Q-�.       ��W�	��&̩�A�* 

Average reward per step�ع��}�       ��2	w�&̩�A�*

epsilon�ع���Pg.       ��W�	5&̩�A�* 

Average reward per step�ع�:T�R       ��2	 6&̩�A�*

epsilon�ع���1.       ��W�	�x&̩�A�* 

Average reward per step�ع��a�       ��2	�y&̩�A�*

epsilon�ع����.       ��W�	a�&̩�A�* 

Average reward per step�ع�4       ��2	L�&̩�A�*

epsilon�ع�_�Pn.       ��W�	�(&̩�A�* 

Average reward per step�ع�&�p       ��2	�)&̩�A�*

epsilon�ع�RrBO.       ��W�	�5&̩�A�* 

Average reward per step�ع�5���       ��2	�6&̩�A�*

epsilon�ع�0(�=.       ��W�	X9&̩�A�* 

Average reward per step�ع��k       ��2	;:&̩�A�*

epsilon�ع�L_$.       ��W�	w&̩�A�* 

Average reward per step�ع��<��       ��2	�w&̩�A�*

epsilon�ع���Ί0       ���_	,�&̩�AM*#
!
Average reward per episode�ӈ�2.       ��W�	�&̩�AM*!

total reward per episode  �³��Y.       ��W�	�&̩�A�* 

Average reward per step�ӈ��-[^       ��2	ظ&̩�A�*

epsilon�ӈ����.       ��W�	�&&̩�A�* 

Average reward per step�ӈ��k�       ��2	�'&̩�A�*

epsilon�ӈ���.       ��W�	̚&̩�A�* 

Average reward per step�ӈ�?�i�       ��2	o�&̩�A�*

epsilon�ӈ�{FC�.       ��W�	{� &̩�A�* 

Average reward per step�ӈ�mG�       ��2	a� &̩�A�*

epsilon�ӈ���CW.       ��W�	M�"&̩�A�* 

Average reward per step�ӈ�H�z       ��2	�"&̩�A�*

epsilon�ӈ��}��.       ��W�	�%&̩�A�* 

Average reward per step�ӈ�۰�g       ��2	.%&̩�A�*

epsilon�ӈ�Ȭ��.       ��W�	�,'&̩�A�* 

Average reward per step�ӈ�Hi�       ��2	k-'&̩�A�*

epsilon�ӈ�Ok�.       ��W�	��(&̩�A�* 

Average reward per step�ӈ���ȱ       ��2	��(&̩�A�*

epsilon�ӈ��!��.       ��W�	B+&̩�A�* 

Average reward per step�ӈ��r       ��2	�+&̩�A�*

epsilon�ӈ��:�y.       ��W�	[_-&̩�A�* 

Average reward per step�ӈ�u��       ��2	`-&̩�A�*

epsilon�ӈ���@K.       ��W�	d�.&̩�A�* 

Average reward per step�ӈ����       ��2	��.&̩�A�*

epsilon�ӈ�'�/�.       ��W�	�i1&̩�A�* 

Average reward per step�ӈ�e�1       ��2	nj1&̩�A�*

epsilon�ӈ�ς��.       ��W�	Ց3&̩�A�* 

Average reward per step�ӈ����       ��2	l�3&̩�A�*

epsilon�ӈ���.       ��W�	��4&̩�A�* 

Average reward per step�ӈ��H��       ��2	��4&̩�A�*

epsilon�ӈ�w�.       ��W�	|�7&̩�A�* 

Average reward per step�ӈ��EZ�       ��2	��7&̩�A�*

epsilon�ӈ�\�v.       ��W�	it;&̩�A�* 

Average reward per step�ӈ�Đ�       ��2	Su;&̩�A�*

epsilon�ӈ�2�>�.       ��W�	��=&̩�A�* 

Average reward per step�ӈ���D       ��2	��=&̩�A�*

epsilon�ӈ��\�.       ��W�	;�?&̩�A�* 

Average reward per step�ӈ��:��       ��2	�?&̩�A�*

epsilon�ӈ�q$�E.       ��W�	4�C&̩�A�* 

Average reward per step�ӈ�x���       ��2	��C&̩�A�*

epsilon�ӈ��d.       ��W�	~6F&̩�A�* 

Average reward per step�ӈ�j^b       ��2	m7F&̩�A�*

epsilon�ӈ����.       ��W�	�I&̩�A�* 

Average reward per step�ӈ�Ȧ��       ��2	ǽI&̩�A�*

epsilon�ӈ�z��}.       ��W�	�.L&̩�A�* 

Average reward per step�ӈ�� +l       ��2	b/L&̩�A�*

epsilon�ӈ�Wt1.       ��W�	+�M&̩�A�* 

Average reward per step�ӈ���r�       ��2	��M&̩�A�*

epsilon�ӈ���Ѯ.       ��W�	�*P&̩�A�* 

Average reward per step�ӈ�C�jO       ��2	�+P&̩�A�*

epsilon�ӈ��p'�.       ��W�	YjR&̩�A�* 

Average reward per step�ӈ�e(�       ��2	kR&̩�A�*

epsilon�ӈ�9��.       ��W�	ץT&̩�A�* 

Average reward per step�ӈ����       ��2	��T&̩�A�*

epsilon�ӈ����.       ��W�	��V&̩�A�* 

Average reward per step�ӈ�!��       ��2	d�V&̩�A�*

epsilon�ӈ���^�.       ��W�	6[&̩�A�* 

Average reward per step�ӈ� l�v       ��2	�[&̩�A�*

epsilon�ӈ� P~�.       ��W�	�m^&̩�A�* 

Average reward per step�ӈ�)�       ��2	�n^&̩�A�*

epsilon�ӈ�S�dq.       ��W�	��`&̩�A�* 

Average reward per step�ӈ�T`;       ��2	��`&̩�A�*

epsilon�ӈ����.       ��W�	�ub&̩�A�* 

Average reward per step�ӈ����       ��2	�vb&̩�A�*

epsilon�ӈ����=.       ��W�	�d&̩�A�* 

Average reward per step�ӈ�<�`A       ��2	��d&̩�A�*

epsilon�ӈ�q��.       ��W�	D�f&̩�A�* 

Average reward per step�ӈ�LP�       ��2	�f&̩�A�*

epsilon�ӈ�m�/}.       ��W�	vi&̩�A�* 

Average reward per step�ӈ�n���       ��2	Pi&̩�A�*

epsilon�ӈ���6l.       ��W�	��j&̩�A�* 

Average reward per step�ӈ��Z�>       ��2	��j&̩�A�*

epsilon�ӈ��^i�.       ��W�	�+m&̩�A�* 

Average reward per step�ӈ�t0��       ��2	s,m&̩�A�*

epsilon�ӈ�1I�.       ��W�	U�n&̩�A�* 

Average reward per step�ӈ�J�       ��2	@�n&̩�A�*

epsilon�ӈ��v0.       ��W�	dq&̩�A�* 

Average reward per step�ӈ�5<I       ��2	q&̩�A�*

epsilon�ӈ�Zۊ:.       ��W�	@�s&̩�A�* 

Average reward per step�ӈ�V��y       ��2	�s&̩�A�*

epsilon�ӈ��<�.       ��W�	��t&̩�A�* 

Average reward per step�ӈ�o^�*       ��2	v�t&̩�A�*

epsilon�ӈ�wog}.       ��W�	C<w&̩�A�* 

Average reward per step�ӈ�p89(       ��2	6=w&̩�A�*

epsilon�ӈ�|T��.       ��W�	Y�y&̩�A�* 

Average reward per step�ӈ��u0#       ��2	�y&̩�A�*

epsilon�ӈ��`n.       ��W�	:?{&̩�A�* 

Average reward per step�ӈ���       ��2	�?{&̩�A�*

epsilon�ӈ��S�2.       ��W�	�}&̩�A�* 

Average reward per step�ӈ�w�T       ��2	��}&̩�A�*

epsilon�ӈ�.v�.       ��W�	�B&̩�A�* 

Average reward per step�ӈ��O�       ��2	NC&̩�A�*

epsilon�ӈ�9#�.       ��W�	���&̩�A�* 

Average reward per step�ӈ����\       ��2	ᛁ&̩�A�*

epsilon�ӈ��ў�.       ��W�	~Ń&̩�A�* 

Average reward per step�ӈ���       ��2	Cƃ&̩�A�*

epsilon�ӈ�C��;.       ��W�	j�&̩�A�* 

Average reward per step�ӈ����-       ��2	H�&̩�A�*

epsilon�ӈ��b!�.       ��W�	F��&̩�A�* 

Average reward per step�ӈ�)~	       ��2	��&̩�A�*

epsilon�ӈ���.       ��W�	W�&̩�A�* 

Average reward per step�ӈ���       ��2	��&̩�A�*

epsilon�ӈ���q.       ��W�	~W�&̩�A�* 

Average reward per step�ӈ��φZ       ��2	\X�&̩�A�*

epsilon�ӈ��ܟK.       ��W�	䢐&̩�A�* 

Average reward per step�ӈ�5��\       ��2	ף�&̩�A�*

epsilon�ӈ���"`.       ��W�	,e�&̩�A�* 

Average reward per step�ӈ��m       ��2	�f�&̩�A�*

epsilon�ӈ�&;��.       ��W�	�&̩�A�* 

Average reward per step�ӈ�7�V       ��2	�&̩�A�*

epsilon�ӈ�d���.       ��W�	w��&̩�A�* 

Average reward per step�ӈ�����       ��2	U��&̩�A�*

epsilon�ӈ�S�R@.       ��W�	~�&̩�A�* 

Average reward per step�ӈ�v�Mw       ��2	u�&̩�A�*

epsilon�ӈ�]/ޢ.       ��W�	�=�&̩�A�* 

Average reward per step�ӈ�q�       ��2	�>�&̩�A�*

epsilon�ӈ�eD�c.       ��W�	���&̩�A�* 

Average reward per step�ӈ��-	&       ��2	���&̩�A�*

epsilon�ӈ� �O.       ��W�	r�&̩�A�* 

Average reward per step�ӈ�Srı       ��2	'�&̩�A�*

epsilon�ӈ����.       ��W�	��&̩�A�* 

Average reward per step�ӈ����       ��2	b�&̩�A�*

epsilon�ӈ�_�O.       ��W�	�k�&̩�A�* 

Average reward per step�ӈ�R�n�       ��2	�l�&̩�A�*

epsilon�ӈ�-8�.       ��W�	Ǡ�&̩�A�* 

Average reward per step�ӈ���'       ��2	j��&̩�A�*

epsilon�ӈ�9ſ�.       ��W�	V�&̩�A�* 

Average reward per step�ӈ���       ��2	W�&̩�A�*

epsilon�ӈ���x.       ��W�	�c�&̩�A�* 

Average reward per step�ӈ�J��       ��2	�d�&̩�A�*

epsilon�ӈ�ag8.       ��W�	�Ӱ&̩�A�* 

Average reward per step�ӈ��S�;       ��2	E԰&̩�A�*

epsilon�ӈ��~Ӹ.       ��W�	|�&̩�A�* 

Average reward per step�ӈ���       ��2	=�&̩�A�*

epsilon�ӈ�����.       ��W�	F�&̩�A�* 

Average reward per step�ӈ�AI�Q       ��2	�F�&̩�A�*

epsilon�ӈ�!�1�.       ��W�	�j�&̩�A�* 

Average reward per step�ӈ�=���       ��2	�k�&̩�A�*

epsilon�ӈ�0�.       ��W�	� �&̩�A�* 

Average reward per step�ӈ�oҊ       ��2	~�&̩�A�*

epsilon�ӈ�V�$^.       ��W�	�I�&̩�A�* 

Average reward per step�ӈ�~�       ��2	IJ�&̩�A�*

epsilon�ӈ�2�q.       ��W�	�'̩�A�* 

Average reward per step�ӈ��\�       ��2	�'̩�A�*

epsilon�ӈ��W*~.       ��W�	�D'̩�A�* 

Average reward per step�ӈ���Ls       ��2	tE'̩�A�*

epsilon�ӈ���.       ��W�	�T'̩�A�* 

Average reward per step�ӈ��]��       ��2	�U'̩�A�*

epsilon�ӈ��13�.       ��W�	��'̩�A�* 

Average reward per step�ӈ�F�m       ��2	��'̩�A�*

epsilon�ӈ�h�%P.       ��W�	�N	'̩�A�* 

Average reward per step�ӈ��Q�&       ��2	�O	'̩�A�*

epsilon�ӈ�.�v�0       ���_	��	'̩�AN*#
!
Average reward per episodeO�'��9.       ��W�	Y�	'̩�AN*!

total reward per episode  �Y��.       ��W�	L�'̩�A�* 

Average reward per stepO�}W&�       ��2	2�'̩�A�*

epsilonO辆M�.       ��W�	} '̩�A�* 

Average reward per stepO�u�t       ��2	!'̩�A�*

epsilonO�~�8�.       ��W�	Z~'̩�A�* 

Average reward per stepO�T�v�       ��2	�~'̩�A�*

epsilonO�\�w4.       ��W�	<�'̩�A�* 

Average reward per stepO�p9�       ��2	ۤ'̩�A�*

epsilonO�{��.       ��W�	�'̩�A�* 

Average reward per stepO���=       ��2	�'̩�A�*

epsilonO辸��.       ��W�	��'̩�A�* 

Average reward per stepO込I�z       ��2	��'̩�A�*

epsilonO辂Zm�.       ��W�	�$'̩�A�* 

Average reward per stepO�����       ��2	�%'̩�A�*

epsilonO�D��.       ��W�	vP '̩�A�* 

Average reward per stepO�'YD�       ��2	+Q '̩�A�*

epsilonO辝�L0.       ��W�	��$'̩�A�* 

Average reward per stepO�D\h       ��2	�$'̩�A�*

epsilonO辟��^.       ��W�	�:('̩�A�* 

Average reward per stepO辝��x       ��2	y;('̩�A�*

epsilonO�a`:�.       ��W�	�*'̩�A�* 

Average reward per stepO�R���       ��2	�*'̩�A�*

epsilonO辦�r.       ��W�	n5,'̩�A�* 

Average reward per stepO��?)�       ��2	D6,'̩�A�*

epsilonO�얢..       ��W�	߈.'̩�A�* 

Average reward per stepO�	�(       ��2	r�.'̩�A�*

epsilonO�q�z�.       ��W�	��0'̩�A�* 

Average reward per stepO�̱�       ��2	d�0'̩�A�*

epsilonO�Qm��.       ��W�	D�2'̩�A�* 

Average reward per stepO辍��       ��2	/�2'̩�A�*

epsilonO辞ښ�.       ��W�	�4'̩�A�* 

Average reward per stepO边��V       ��2	��4'̩�A�*

epsilonO辀ӻZ.       ��W�	eo6'̩�A�* 

Average reward per stepO�����       ��2	ap6'̩�A�*

epsilonO�z�`/.       ��W�	�8'̩�A�* 

Average reward per stepO辨�/�       ��2	(�8'̩�A�*

epsilonO辖5�.       ��W�	5&;'̩�A�* 

Average reward per stepO�)v��       ��2	�&;'̩�A�*

epsilonO�@Xb.       ��W�	L7='̩�A�* 

Average reward per stepO辀��}       ��2	8='̩�A�*

epsilonO�w���.       ��W�	�>'̩�A�* 

Average reward per stepO���r       ��2	��>'̩�A�*

epsilonO�b�(�.       ��W�	"A'̩�A�* 

Average reward per stepO�J��       ��2	�A'̩�A�*

epsilonO�!��8.       ��W�	��B'̩�A�* 

Average reward per stepO�=��\       ��2	t�B'̩�A�*

epsilonO�)R�	.       ��W�	�	E'̩�A�* 

Average reward per stepO�ӯ'f       ��2	l
E'̩�A�*

epsilonO�?�Z.       ��W�	H4G'̩�A�* 

Average reward per stepO��T��       ��2	?5G'̩�A�*

epsilonO辏�b.       ��W�	�I'̩�A�* 

Average reward per stepO�a�~w       ��2	�I'̩�A�*

epsilonO��S�.       ��W�	�L'̩�A�* 

Average reward per stepO�-g�       ��2	��L'̩�A�*

epsilonO辍\.       ��W�	-%O'̩�A�* 

Average reward per stepO达Rv�       ��2	�%O'̩�A�*

epsilonO辴TW�0       ���_	NO'̩�AO*#
!
Average reward per episode�$��[C|p.       ��W�	�NO'̩�AO*!

total reward per episode  ñ�:Z.       ��W�	��S'̩�A�* 

Average reward per step�$���S�"       ��2	��S'̩�A�*

epsilon�$����.       ��W�	�iU'̩�A�* 

Average reward per step�$���Ċ       ��2	�jU'̩�A�*

epsilon�$��J�_N.       ��W�	��W'̩�A�* 

Average reward per step�$��[ğ�       ��2	� X'̩�A�*

epsilon�$���yG.       ��W�	��['̩�A�* 

Average reward per step�$���Y�g       ��2	F�['̩�A�*

epsilon�$��[�Aa.       ��W�	��]'̩�A�* 

Average reward per step�$��eT�       ��2	��]'̩�A�*

epsilon�$��4$��.       ��W�	��`'̩�A�* 

Average reward per step�$��(���       ��2	�`'̩�A�*

epsilon�$����8.       ��W�	�c'̩�A�* 

Average reward per step�$���z�0       ��2	��c'̩�A�*

epsilon�$���wE.       ��W�	��f'̩�A�* 

Average reward per step�$��u��       ��2	?�f'̩�A�*

epsilon�$���)$�.       ��W�	#Lj'̩�A�* 

Average reward per step�$��,z       ��2	�Lj'̩�A�*

epsilon�$��]0*y.       ��W�	{l'̩�A�* 

Average reward per step�$��w!�;       ��2	{l'̩�A�*

epsilon�$���j+�.       ��W�	DMn'̩�A�* 

Average reward per step�$����Q�       ��2	�Mn'̩�A�*

epsilon�$��k(!.       ��W�	�pp'̩�A�* 

Average reward per step�$���       ��2	�qp'̩�A�*

epsilon�$���B�2.       ��W�	�r'̩�A�* 

Average reward per step�$��q^�       ��2	f�r'̩�A�*

epsilon�$��32�?.       ��W�	�wt'̩�A�* 

Average reward per step�$����T�       ��2	�xt'̩�A�*

epsilon�$��ҝ��.       ��W�	]Nw'̩�A�* 

Average reward per step�$��1���       ��2	<Ow'̩�A�*

epsilon�$���ה.       ��W�	�z'̩�A�* 

Average reward per step�$��f�       ��2	�z'̩�A�*

epsilon�$���\R.       ��W�	�}'̩�A�* 

Average reward per step�$����       ��2	�}'̩�A�*

epsilon�$�����e.       ��W�	8f�'̩�A�* 

Average reward per step�$��c���       ��2	�f�'̩�A�*

epsilon�$���=��.       ��W�	(|�'̩�A�* 

Average reward per step�$��?Z��       ��2	�|�'̩�A�*

epsilon�$��Թ8.       ��W�	�*�'̩�A�* 

Average reward per step�$�����h       ��2	�,�'̩�A�*

epsilon�$��Zt�+.       ��W�	�>�'̩�A�* 

Average reward per step�$��앭�       ��2	�?�'̩�A�*

epsilon�$��b��.       ��W�	�m�'̩�A�* 

Average reward per step�$��`��y       ��2	�n�'̩�A�*

epsilon�$��&���.       ��W�	���'̩�A�* 

Average reward per step�$���5�       ��2	{��'̩�A�*

epsilon�$��Xfw�.       ��W�	>]�'̩�A�* 

Average reward per step�$��7��       ��2	N^�'̩�A�*

epsilon�$��e���.       ��W�	"6�'̩�A�* 

Average reward per step�$�����9       ��2	�6�'̩�A�*

epsilon�$��E��0       ���_	�R�'̩�AP*#
!
Average reward per episode\����N�.       ��W�	TS�'̩�AP*!

total reward per episode  �aHD�.       ��W�	C�'̩�A�* 

Average reward per step\����y       ��2	�D�'̩�A�*

epsilon\���<]�k.       ��W�	r��'̩�A�* 

Average reward per step\���}�?_       ��2	��'̩�A�*

epsilon\���]�o8.       ��W�	�'̩�A�* 

Average reward per step\����J�       ��2	 �'̩�A�*

epsilon\�����LM.       ��W�	(E�'̩�A�* 

Average reward per step\����y��       ��2	,F�'̩�A�*

epsilon\����s.       ��W�	%z�'̩�A�* 

Average reward per step\����y�       ��2	{�'̩�A�*

epsilon\���� �.       ��W�		��'̩�A�* 

Average reward per step\����T       ��2	ڍ�'̩�A�*

epsilon\������.       ��W�	�U�'̩�A�* 

Average reward per step\���$��H       ��2	�V�'̩�A�*

epsilon\����'�K.       ��W�	��'̩�A�* 

Average reward per step\���4.�       ��2	���'̩�A�*

epsilon\����	�.       ��W�	�s�'̩�A�* 

Average reward per step\����<%       ��2	�t�'̩�A�*

epsilon\���V��.       ��W�	*��'̩�A�* 

Average reward per step\���O��0       ��2	ސ�'̩�A�*

epsilon\���!Ӽ0.       ��W�	К�'̩�A�* 

Average reward per step\���(�=�       ��2	���'̩�A�*

epsilon\����/�E.       ��W�	��'̩�A�* 

Average reward per step\����-��       ��2	���'̩�A�*

epsilon\����ܛ�.       ��W�	�M�'̩�A�* 

Average reward per step\�����l       ��2	�N�'̩�A�*

epsilon\���s���.       ��W�	���'̩�A�* 

Average reward per step\����7�@       ��2	L��'̩�A�*

epsilon\���,0�.       ��W�	��'̩�A�* 

Average reward per step\���1�-J       ��2	��'̩�A�*

epsilon\���t��.       ��W�	&�(̩�A�* 

Average reward per step\���<��~       ��2	�(̩�A�*

epsilon\����s��.       ��W�	j4(̩�A�* 

Average reward per step\���'�-�       ��2	i5(̩�A�*

epsilon\������l.       ��W�	��(̩�A�* 

Average reward per step\����/�       ��2	��(̩�A�*

epsilon\���I:q�.       ��W�	X<(̩�A�* 

Average reward per step\����[�       ��2	h=(̩�A�*

epsilon\���3�J.       ��W�	1�(̩�A�* 

Average reward per step\���~C�       ��2	��(̩�A�*

epsilon\������g.       ��W�	�C(̩�A�* 

Average reward per step\�������       ��2	�D(̩�A�*

epsilon\�����M�.       ��W�	]o(̩�A�* 

Average reward per step\���>C~�       ��2	*p(̩�A�*

epsilon\����4��.       ��W�	(̩�A�* 

Average reward per step\����ɔ�       ��2	(̩�A�*

epsilon\���N��.       ��W�	=d(̩�A�* 

Average reward per step\���*L�X       ��2	ge(̩�A�*

epsilon\���Q�P�.       ��W�	�(̩�A�* 

Average reward per step\���M�ta       ��2	̳(̩�A�*

epsilon\����b�O.       ��W�	 t(̩�A�* 

Average reward per step\����ٜ�       ��2	�t(̩�A�*

epsilon\����O�L.       ��W�	D�(̩�A�* 

Average reward per step\����+��       ��2	7�(̩�A�*

epsilon\����.9.       ��W�	�K(̩�A�* 

Average reward per step\�����@�       ��2	�L(̩�A�*

epsilon\����Sn.       ��W�	&�!(̩�A�* 

Average reward per step\���nu�       ��2	��!(̩�A�*

epsilon\���ۉ;.       ��W�	�b$(̩�A�* 

Average reward per step\�����t       ��2	Vc$(̩�A�*

epsilon\���d.c.       ��W�	��'(̩�A�* 

Average reward per step\����%�       ��2	v�'(̩�A�*

epsilon\����ȎY.       ��W�	>�)(̩�A�* 

Average reward per step\���
       ��2	�)(̩�A�*

epsilon\���O9w.       ��W�	`?,(̩�A�* 

Average reward per step\���mp7A       ��2	:@,(̩�A�*

epsilon\������?.       ��W�	�/(̩�A�* 

Average reward per step\���;        ��2	��/(̩�A�*

epsilon\���4�-0       ���_	�	0(̩�AQ*#
!
Average reward per episode�҂�MFA.       ��W�	9
0(̩�AQ*!

total reward per episode  ��S�3.       ��W�	�3(̩�A�* 

Average reward per step�҂�^%/       ��2	#�3(̩�A�*

epsilon�҂��od�.       ��W�	��6(̩�A�* 

Average reward per step�҂���       ��2	M�6(̩�A�*

epsilon�҂�8�<.       ��W�	�C:(̩�A�* 

Average reward per step�҂���Ǥ       ��2	=E:(̩�A�*

epsilon�҂�a��n.       ��W�	�m<(̩�A�* 

Average reward per step�҂�0i�       ��2	�n<(̩�A�*

epsilon�҂�>{}.       ��W�	>(̩�A�* 

Average reward per step�҂��{R�       ��2	�>(̩�A�*

epsilon�҂��=�.       ��W�	>[@(̩�A�* 

Average reward per step�҂��:q�       ��2	\@(̩�A�*

epsilon�҂�@)�5.       ��W�	u�B(̩�A�* 

Average reward per step�҂��o�I       ��2	h�B(̩�A�*

epsilon�҂��F/.       ��W�	�ED(̩�A�* 

Average reward per step�҂�7n�       ��2	�FD(̩�A�*

epsilon�҂�Z�.�.       ��W�	i�F(̩�A�* 

Average reward per step�҂�UL��       ��2	�F(̩�A�*

epsilon�҂���F\.       ��W�	d\H(̩�A�* 

Average reward per step�҂�����       ��2	y]H(̩�A�*

epsilon�҂��~T.       ��W�	�J(̩�A�* 

Average reward per step�҂�7q       ��2	ޓJ(̩�A�*

epsilon�҂��f6�.       ��W�	K�L(̩�A�* 

Average reward per step�҂�{Bv       ��2	�L(̩�A�*

epsilon�҂�2��.       ��W�	�N(̩�A�* 

Average reward per step�҂���       ��2	#�N(̩�A�*

epsilon�҂�����.       ��W�	g�P(̩�A�* 

Average reward per step�҂��l�*       ��2	�P(̩�A�*

epsilon�҂��`'�.       ��W�	?RS(̩�A�* 

Average reward per step�҂�%�7       ��2	7SS(̩�A�*

epsilon�҂��O�P.       ��W�	�[W(̩�A�* 

Average reward per step�҂�d2v       ��2	]W(̩�A�*

epsilon�҂�OS-.       ��W�	r�Z(̩�A�* 

Average reward per step�҂�J�P[       ��2	v�Z(̩�A�*

epsilon�҂��3y�.       ��W�	�](̩�A�* 

Average reward per step�҂�X���       ��2	S	](̩�A�*

epsilon�҂�yȂ�.       ��W�	xc_(̩�A�* 

Average reward per step�҂��A�       ��2	Vd_(̩�A�*

epsilon�҂�Q��K.       ��W�	8�a(̩�A�* 

Average reward per step�҂�D��       ��2	��a(̩�A�*

epsilon�҂�=���.       ��W�	�b(̩�A�* 

Average reward per step�҂����1       ��2	��b(̩�A�*

epsilon�҂�7	[~0       ���_	�>c(̩�AR*#
!
Average reward per episode�m���U�.       ��W�	F?c(̩�AR*!

total reward per episode  �P�.       ��W�	��g(̩�A�* 

Average reward per step�m�����       ��2	��g(̩�A�*

epsilon�m���.       ��W�	�Yk(̩�A�* 

Average reward per step�m��I 4�       ��2	OZk(̩�A�*

epsilon�m���څ�.       ��W�	��m(̩�A�* 

Average reward per step�m�����       ��2	��m(̩�A�*

epsilon�m��j��g.       ��W�	;oo(̩�A�* 

Average reward per step�m����-       ��2	po(̩�A�*

epsilon�m��k���.       ��W�	ӈq(̩�A�* 

Average reward per step�m��Dљ�       ��2	q(̩�A�*

epsilon�m��$V�.       ��W�	�s(̩�A�* 

Average reward per step�m��U��       ��2	��s(̩�A�*

epsilon�m���`s�.       ��W�	�"v(̩�A�* 

Average reward per step�m���j�h       ��2	W#v(̩�A�*

epsilon�m��E�#�.       ��W�	Vx(̩�A�* 

Average reward per step�m��<�WT       ��2	Ex(̩�A�*

epsilon�m��
T�.       ��W�	��z(̩�A�* 

Average reward per step�m��=d       ��2	��z(̩�A�*

epsilon�m�����B.       ��W�	�P~(̩�A�* 

Average reward per step�m��f���       ��2	�Q~(̩�A�*

epsilon�m��:ߌp.       ��W�	 �(̩�A�* 

Average reward per step�m���m�       ��2	��(̩�A�*

epsilon�m���U�.       ��W�	�G�(̩�A�* 

Average reward per step�m����y�       ��2	EH�(̩�A�*

epsilon�m���Nl.       ��W�	;��(̩�A�* 

Average reward per step�m��|`�\       ��2	*�(̩�A�*

epsilon�m����uV.       ��W�	�2�(̩�A�* 

Average reward per step�m���s��       ��2	�3�(̩�A�*

epsilon�m���#��.       ��W�	mȈ(̩�A�* 

Average reward per step�m��2 �|       ��2	Ɉ(̩�A�*

epsilon�m��� �].       ��W�	Hl�(̩�A�* 

Average reward per step�m��v��       ��2	/m�(̩�A�*

epsilon�m��Riv.       ��W�	d��(̩�A�* 

Average reward per step�m��V��K       ��2	��(̩�A�*

epsilon�m���#�9.       ��W�	�(̩�A�* 

Average reward per step�m����M       ��2	��(̩�A�*

epsilon�m��8B).       ��W�	n�(̩�A�* 

Average reward per step�m����E	       ��2	��(̩�A�*

epsilon�m��1���.       ��W�	�U�(̩�A�* 

Average reward per step�m���3�       ��2	`V�(̩�A�*

epsilon�m��"��.       ��W�	/��(̩�A�* 

Average reward per step�m���$.       ��2	���(̩�A�*

epsilon�m��Y�`.       ��W�	�8�(̩�A�* 

Average reward per step�m����4s       ��2	�9�(̩�A�*

epsilon�m��y(6.       ��W�	F[�(̩�A�* 

Average reward per step�m��XB�       ��2	)\�(̩�A�*

epsilon�m��p��}.       ��W�	���(̩�A�* 

Average reward per step�m���/@       ��2	ʣ�(̩�A�*

epsilon�m���/m0       ���_	]¡(̩�AS*#
!
Average reward per episode  ���eu.       ��W�	�¡(̩�AS*!

total reward per episode  ���=�.       ��W�		k�(̩�A�* 

Average reward per step  ����G        ��2	�k�(̩�A�*

epsilon  ������.       ��W�	���(̩�A�* 

Average reward per step  ��_��       ��2	Υ�(̩�A�*

epsilon  ��pn}�.       ��W�	�N�(̩�A�* 

Average reward per step  ���5�#       ��2	�O�(̩�A�*

epsilon  ��T"(.       ��W�	㈭(̩�A�* 

Average reward per step  ���{�       ��2	���(̩�A�*

epsilon  ��y2�.       ��W�	kү(̩�A�* 

Average reward per step  ����kI       ��2	Rӯ(̩�A�*

epsilon  ��zȹ.       ��W�	�f�(̩�A�* 

Average reward per step  ����       ��2	�g�(̩�A�*

epsilon  ��� &�.       ��W�	���(̩�A�* 

Average reward per step  ��;�       ��2	���(̩�A�*

epsilon  ����i.       ��W�	ŵ(̩�A�* 

Average reward per step  ��>�Z       ��2	�ŵ(̩�A�*

epsilon  ��ͥi�.       ��W�	��(̩�A�* 

Average reward per step  ��Z3�
       ��2	��(̩�A�*

epsilon  ���
.       ��W�	���(̩�A�* 

Average reward per step  ��N��D       ��2	���(̩�A�*

epsilon  ������.       ��W�	F��(̩�A�* 

Average reward per step  ��9,@3       ��2	���(̩�A�*

epsilon  ��6N.�.       ��W�	C�(̩�A�* 

Average reward per step  ��
,)�       ��2	�C�(̩�A�*

epsilon  ��r/�.       ��W�	F��(̩�A�* 

Average reward per step  �����       ��2	�(̩�A�*

epsilon  �����.       ��W�	,�(̩�A�* 

Average reward per step  ����b�       ��2	�,�(̩�A�*

epsilon  ���I/�.       ��W�	E��(̩�A�* 

Average reward per step  ��]�-m       ��2	��(̩�A�*

epsilon  ����נ.       ��W�	�S�(̩�A�* 

Average reward per step  ��s��       ��2	�T�(̩�A�*

epsilon  �����'.       ��W�	M��(̩�A�* 

Average reward per step  ���(q       ��2	��(̩�A�*

epsilon  ��ųO�.       ��W�	W��(̩�A�* 

Average reward per step  ��d��       ��2	-��(̩�A�*

epsilon  ��v�.       ��W�	��(̩�A�* 

Average reward per step  ��.H�       ��2	}�(̩�A�*

epsilon  ��#�n.       ��W�	��(̩�A�* 

Average reward per step  ��"��'       ��2	��(̩�A�*

epsilon  ����^�.       ��W�	b.�(̩�A�* 

Average reward per step  ���xv�       ��2	j/�(̩�A�*

epsilon  ��k;+.       ��W�	o�(̩�A�* 

Average reward per step  ���T��       ��2	�(̩�A�*

epsilon  ��bs �.       ��W�	c�(̩�A�* 

Average reward per step  ��_<��       ��2	�c�(̩�A�*

epsilon  ���奛.       ��W�	L��(̩�A�* 

Average reward per step  ��B�       ��2	; �(̩�A�*

epsilon  ��N{��.       ��W�	�=�(̩�A�* 

Average reward per step  �����#       ��2	G>�(̩�A�*

epsilon  �����.       ��W�	�j�(̩�A�* 

Average reward per step  ��)��
       ��2	�k�(̩�A�*

epsilon  �����.       ��W�	Kx�(̩�A�* 

Average reward per step  ���X�-       ��2	dy�(̩�A�*

epsilon  ����.       ��W�	���(̩�A�* 

Average reward per step  ��|BZ       ��2	"��(̩�A�*

epsilon  ������.       ��W�	���(̩�A�* 

Average reward per step  ��(s�       ��2	k��(̩�A�*

epsilon  �����P.       ��W�	�3�(̩�A�* 

Average reward per step  ���1d~       ��2	v4�(̩�A�*

epsilon  ��A���.       ��W�	7��(̩�A�* 

Average reward per step  �����       ��2	��(̩�A�*

epsilon  ���}�.       ��W�	��(̩�A�* 

Average reward per step  ���uh       ��2	��(̩�A�*

epsilon  ��mx�.       ��W�	���(̩�A�* 

Average reward per step  ��V��0       ��2	���(̩�A�*

epsilon  ����5�.       ��W�	��(̩�A�* 

Average reward per step  ���"<�       ��2	Ȕ�(̩�A�*

epsilon  ������.       ��W�	j� )̩�A�* 

Average reward per step  �����       ��2	M� )̩�A�*

epsilon  ������.       ��W�	��)̩�A�* 

Average reward per step  ������       ��2	�)̩�A�*

epsilon  ��5+�.       ��W�	�)̩�A�* 

Average reward per step  ���Q�       ��2	�)̩�A�*

epsilon  �����.       ��W�	�	)̩�A�* 

Average reward per step  ��(6Wb       ��2	�	)̩�A�*

epsilon  ��p�u0.       ��W�	D�)̩�A�* 

Average reward per step  ��"�s�       ��2	z�)̩�A�*

epsilon  ����4.       ��W�	�O)̩�A�* 

Average reward per step  ��y'�3       ��2	PQ)̩�A�*

epsilon  ���^��.       ��W�	�])̩�A�* 

Average reward per step  ���d�V       ��2	_)̩�A�*

epsilon  ��(T��.       ��W�	@N)̩�A�* 

Average reward per step  ����H       ��2	�O)̩�A�*

epsilon  ��f|a�.       ��W�	II)̩�A�* 

Average reward per step  ��Kf       ��2	�J)̩�A�*

epsilon  ��#�N�.       ��W�	�E)̩�A�* 

Average reward per step  ���=�       ��2	�F)̩�A�*

epsilon  ���'ʓ.       ��W�	��)̩�A�* 

Average reward per step  ��@dW       ��2	��)̩�A�*

epsilon  ��O�5�.       ��W�	X�#)̩�A�* 

Average reward per step  ��a���       ��2	S�#)̩�A�*

epsilon  ���(� .       ��W�	Z�')̩�A�* 

Average reward per step  �����q       ��2	8�')̩�A�*

epsilon  ����tB0       ���_	��')̩�AT*#
!
Average reward per episode   ��K>�.       ��W�	Y�')̩�AT*!

total reward per episode  ��=Q �.       ��W�	�-)̩�A�* 

Average reward per step   �Ee��       ��2	��-)̩�A�*

epsilon   ��K�6.       ��W�	B�1)̩�A�* 

Average reward per step   �T�M       ��2	W�1)̩�A�*

epsilon   �\xd�.       ��W�	*�5)̩�A�* 

Average reward per step   �p���       ��2	i�5)̩�A�*

epsilon   �f[f.       ��W�	4g8)̩�A�* 

Average reward per step   �$n�}       ��2	Uh8)̩�A�*

epsilon   ���|^.       ��W�	$�;)̩�A�* 

Average reward per step   �=��w       ��2	I�;)̩�A�*

epsilon   ����.       ��W�	?>)̩�A�* 

Average reward per step   ��*u       ��2	@>)̩�A�*

epsilon   ����.       ��W�	�@)̩�A�* 

Average reward per step   ��䎚       ��2	�@)̩�A�*

epsilon   �EXW�.       ��W�	y�D)̩�A�* 

Average reward per step   �����       ��2	��D)̩�A�*

epsilon   ��a��.       ��W�	�xH)̩�A�* 

Average reward per step   ��� �       ��2	hyH)̩�A�*

epsilon   ��y�%.       ��W�	(IJ)̩�A�* 

Average reward per step   �����       ��2	AJJ)̩�A�*

epsilon   �
��.       ��W�	�(L)̩�A�* 

Average reward per step   ��q��       ��2	x)L)̩�A�*

epsilon   �r�8,.       ��W�	�N)̩�A�* 

Average reward per step   ���L�       ��2	��N)̩�A�*

epsilon   �A�b�.       ��W�	ofQ)̩�A�* 

Average reward per step   ����       ��2	ZgQ)̩�A�*

epsilon   �e�e.       ��W�	V)̩�A�* 

Average reward per step   �H��       ��2	AV)̩�A�*

epsilon   ����s.       ��W�	X�Y)̩�A�* 

Average reward per step   �maن       ��2	��Y)̩�A�*

epsilon   �'���.       ��W�	�G])̩�A�* 

Average reward per step   ��_M       ��2	0I])̩�A�*

epsilon   ��]& .       ��W�	��_)̩�A�* 

Average reward per step   ��k#�       ��2	u�_)̩�A�*

epsilon   ����.       ��W�	S�c)̩�A�* 

Average reward per step   �p�       ��2	��c)̩�A�*

epsilon   ��@�0       ���_	�c)̩�AU*#
!
Average reward per episode�������.       ��W�	��c)̩�AU*!

total reward per episode  #����.       ��W�	��i)̩�A�* 

Average reward per step�����N       ��2	i�i)̩�A�*

epsilon���Vͩ�.       ��W�	�0l)̩�A�* 

Average reward per step�����"       ��2	82l)̩�A�*

epsilon���`�.       ��W�	�p)̩�A�* 

Average reward per step���Zb�e       ��2	 p)̩�A�*

epsilon���d��.       ��W�	��r)̩�A�* 

Average reward per step���xo�s       ��2	��r)̩�A�*

epsilon���<a��.       ��W�	X�v)̩�A�* 

Average reward per step������       ��2	��v)̩�A�*

epsilon�����e.       ��W�	S�z)̩�A�* 

Average reward per step�����       ��2	)�z)̩�A�*

epsilon����Έ�.       ��W�	�~)̩�A�* 

Average reward per step���C1f*       ��2	 �~)̩�A�*

epsilon����u�.       ��W�	��)̩�A�* 

Average reward per step������v       ��2	.��)̩�A�*

epsilon���HZ��.       ��W�	�)̩�A�* 

Average reward per step����ϑ�       ��2	DĄ)̩�A�*

epsilon�����,.       ��W�	�@�)̩�A�* 

Average reward per step���LK�       ��2	�A�)̩�A�*

epsilon����_~�.       ��W�	u�)̩�A�* 

Average reward per step���#צ�       ��2	��)̩�A�*

epsilon�����.       ��W�	ط�)̩�A�* 

Average reward per step�����;6       ��2	A��)̩�A�*

epsilon���tc�y.       ��W�	�s�)̩�A�* 

Average reward per step������       ��2	u�)̩�A�*

epsilon����JQ=.       ��W�	\�)̩�A�* 

Average reward per step�����%       ��2	m�)̩�A�*

epsilon���?��".       ��W�	¤�)̩�A�* 

Average reward per step�����       ��2		��)̩�A�*

epsilon����Wp�.       ��W�	�W�)̩�A�* 

Average reward per step�����Ԛ       ��2	Y�)̩�A�*

epsilon�����oS.       ��W�	��)̩�A�* 

Average reward per step�����       ��2	^��)̩�A�*

epsilon���o� M.       ��W�	#I�)̩�A�* 

Average reward per step����h6�       ��2	IJ�)̩�A�*

epsilon����+�-.       ��W�	��)̩�A�* 

Average reward per step����]�^       ��2	��)̩�A�*

epsilon�����.       ��W�	�~�)̩�A�* 

Average reward per step����]A       ��2	��)̩�A�*

epsilon���_���.       ��W�	@/�)̩�A�* 

Average reward per step���-V       ��2	�0�)̩�A�*

epsilon���tTr.       ��W�	q�)̩�A�* 

Average reward per step����+�V       ��2	��)̩�A�*

epsilon����v��.       ��W�	���)̩�A�* 

Average reward per step����]l�       ��2	���)̩�A�*

epsilon���YZ��.       ��W�	�]�)̩�A�* 

Average reward per step���F7	       ��2	�^�)̩�A�*

epsilon���b,�0       ���_	��)̩�AV*#
!
Average reward per episode�����C��.       ��W�	���)̩�AV*!

total reward per episode  	��t�.       ��W�	�^�)̩�A�* 

Average reward per step�������w       ��2	�_�)̩�A�*

epsilon����<��.       ��W�	I/�)̩�A�* 

Average reward per step�����J�       ��2	n0�)̩�A�*

epsilon������lD.       ��W�	�2�)̩�A�* 

Average reward per step����IH�Z       ��2	�3�)̩�A�*

epsilon�����Qԙ.       ��W�	'��)̩�A�* 

Average reward per step�����;߾       ��2	&��)̩�A�*

epsilon�������.       ��W�	dw�)̩�A�* 

Average reward per step�����i       ��2	lx�)̩�A�*

epsilon������<.       ��W�	���)̩�A�* 

Average reward per step����ݾ�W       ��2	���)̩�A�*

epsilon����C�D�.       ��W�	9��)̩�A�* 

Average reward per step����a��K       ��2	���)̩�A�*

epsilon������@�.       ��W�	;r�)̩�A�* 

Average reward per step����M�-�       ��2	!s�)̩�A�*

epsilon�����	��.       ��W�	�,�)̩�A�* 

Average reward per step����>��       ��2	b-�)̩�A�*

epsilon������O�.       ��W�	���)̩�A�* 

Average reward per step����$��W       ��2	���)̩�A�*

epsilon�������.       ��W�	SA�)̩�A�* 

Average reward per step�����dT       ��2	$B�)̩�A�*

epsilon�����>��.       ��W�	���)̩�A�* 

Average reward per step����Q���       ��2	o��)̩�A�*

epsilon����}}��.       ��W�	-A�)̩�A�* 

Average reward per step����d�bi       ��2	B�)̩�A�*

epsilon���� ��.       ��W�	��)̩�A�* 

Average reward per step�������       ��2	4��)̩�A�*

epsilon�����"�u.       ��W�	��)̩�A�* 

Average reward per step�����V�f       ��2	a��)̩�A�*

epsilon����1�9.       ��W�	���)̩�A�* 

Average reward per step������ 9       ��2	���)̩�A�*

epsilon����xe��.       ��W�	V�)̩�A�* 

Average reward per step��������       ��2	&W�)̩�A�*

epsilon�����m�.       ��W�	&8�)̩�A�* 

Average reward per step�����N�x       ��2	 9�)̩�A�*

epsilon����]�5�0       ���_	<��)̩�AW*#
!
Average reward per episode�8��,��.       ��W�	]��)̩�AW*!

total reward per episode  ���L�.       ��W�	��)̩�A�* 

Average reward per step�8����       ��2	��)̩�A�*

epsilon�8��}��.       ��W�	I��)̩�A�* 

Average reward per step�8��j\       ��2	<��)̩�A�*

epsilon�8��S.       ��W�	mo�)̩�A�* 

Average reward per step�8�0��       ��2	rp�)̩�A�*

epsilon�8�
��.       ��W�	z*̩�A�* 

Average reward per step�8��[0       ��2	P*̩�A�*

epsilon�8����M.       ��W�	��*̩�A�* 

Average reward per step�8�����       ��2	y�*̩�A�*

epsilon�8��[��.       ��W�	]�*̩�A�* 

Average reward per step�8�s5F?       ��2	L�*̩�A�*

epsilon�8�6�j.       ��W�	;4*̩�A�* 

Average reward per step�8�z��       ��2	5*̩�A�*

epsilon�8�a�}�.       ��W�	�*̩�A�* 

Average reward per step�8�3%�       ��2	�*̩�A�*

epsilon�8�����.       ��W�	fl*̩�A�* 

Average reward per step�8�?H�!       ��2	eo*̩�A�*

epsilon�8�UJ�.       ��W�	��*̩�A�* 

Average reward per step�8��8       ��2	�*̩�A�*

epsilon�8���t�.       ��W�	��*̩�A�* 

Average reward per step�8��.�       ��2	7�*̩�A�*

epsilon�8���,I.       ��W�	b�*̩�A�* 

Average reward per step�8��S�       ��2	8�*̩�A�*

epsilon�8�[j9�.       ��W�	\*̩�A�* 

Average reward per step�8��u�       ��2	?*̩�A�*

epsilon�8�y�.�.       ��W�	`�!*̩�A�* 

Average reward per step�8� Qp�       ��2	��!*̩�A�*

epsilon�8��v;[.       ��W�	�+&*̩�A�* 

Average reward per step�8��I       ��2	�,&*̩�A�*

epsilon�8�$�*.       ��W�	Ii)*̩�A�* 

Average reward per step�8�a
��       ��2	/j)*̩�A�*

epsilon�8�U��.       ��W�	��+*̩�A�* 

Average reward per step�8�Q#�       ��2	w�+*̩�A�*

epsilon�8�f�[$.       ��W�	��-*̩�A�* 

Average reward per step�8�2�1�       ��2	��-*̩�A�*

epsilon�8�5@?q.       ��W�	7�2*̩�A�* 

Average reward per step�8���j�       ��2	;�2*̩�A�*

epsilon�8��a|.       ��W�	�a6*̩�A�* 

Average reward per step�8����       ��2	�b6*̩�A�*

epsilon�8���.       ��W�	�:*̩�A�* 

Average reward per step�8��Z�f       ��2	��:*̩�A�*

epsilon�8��c�.       ��W�	��>*̩�A�* 

Average reward per step�8�<��       ��2	�>*̩�A�*

epsilon�8�c�Ӯ.       ��W�	�UC*̩�A�* 

Average reward per step�8��|e�       ��2	KWC*̩�A�*

epsilon�8�	�f�.       ��W�	f�F*̩�A�* 

Average reward per step�8�����       ��2	@�F*̩�A�*

epsilon�8�t���.       ��W�	5aH*̩�A�* 

Average reward per step�8�ܖ�       ��2	AbH*̩�A�*

epsilon�8�v��W.       ��W�	p�J*̩�A�* 

Average reward per step�8�E+�}       ��2	��J*̩�A�*

epsilon�8��EwB.       ��W�	xO*̩�A�* 

Average reward per step�8����       ��2	}O*̩�A�*

epsilon�8��ZО.       ��W�	ߥR*̩�A�* 

Average reward per step�8�i��D       ��2	ߦR*̩�A�*

epsilon�8��K1�.       ��W�	�MU*̩�A�* 

Average reward per step�8�~�1�       ��2	�NU*̩�A�*

epsilon�8��a�.       ��W�	+Y*̩�A�* 

Average reward per step�8�ř,�       ��2	]Y*̩�A�*

epsilon�8�E$��.       ��W�	�6[*̩�A�* 

Average reward per step�8�/���       ��2	78[*̩�A�*

epsilon�8�Ť�.       ��W�	^/_*̩�A�* 

Average reward per step�8��B;�       ��2	�0_*̩�A�*

epsilon�8��[��.       ��W�	4�a*̩�A�* 

Average reward per step�8�����       ��2	D�a*̩�A�*

epsilon�8�1�F�.       ��W�	%f*̩�A�* 

Average reward per step�8��hg       ��2	>&f*̩�A�*

epsilon�8����F0       ���_	DNf*̩�AX*#
!
Average reward per episode��g���!}.       ��W�	/Of*̩�AX*!

total reward per episode  ���?�.       ��W�	� n*̩�A�* 

Average reward per step��g��J�5       ��2	n*̩�A�*

epsilon��g�TU .       ��W�	-�q*̩�A�* 

Average reward per step��g���+       ��2	�q*̩�A�*

epsilon��g���.       ��W�	?t*̩�A�* 

Average reward per step��g����       ��2	�@t*̩�A�*

epsilon��g�`'�a.       ��W�	&Rx*̩�A�* 

Average reward per step��g��C��       ��2	�Rx*̩�A�*

epsilon��g�*��.       ��W�	��|*̩�A�* 

Average reward per step��g�t+�}       ��2	��|*̩�A�*

epsilon��g�^�C.       ��W�	�C�*̩�A�* 

Average reward per step��g���I[       ��2	$E�*̩�A�*

epsilon��g�6�O.       ��W�	R�*̩�A�* 

Average reward per step��g�2jP&       ��2	x�*̩�A�*

epsilon��g�j��.       ��W�	�+�*̩�A�* 

Average reward per step��g��`�G       ��2	�,�*̩�A�*

epsilon��g�\� 7.       ��W�	�9�*̩�A�* 

Average reward per step��g���0z       ��2	*;�*̩�A�*

epsilon��g�����.       ��W�	��*̩�A�* 

Average reward per step��g�Pg��       ��2	��*̩�A�*

epsilon��g��v��.       ��W�	���*̩�A�* 

Average reward per step��g�Դ^�       ��2	���*̩�A�*

epsilon��g���(�.       ��W�	��*̩�A�* 

Average reward per step��g��j?       ��2	��*̩�A�*

epsilon��g���1.       ��W�	~Ė*̩�A�* 

Average reward per step��g�`F��       ��2	�Ŗ*̩�A�*

epsilon��g�t���.       ��W�	�@�*̩�A�* 

Average reward per step��g����       ��2	B�*̩�A�*

epsilon��g�}���.       ��W�	�&�*̩�A�* 

Average reward per step��g��i��       ��2	�'�*̩�A�*

epsilon��g���.       ��W�	Y��*̩�A�* 

Average reward per step��g�'{��       ��2	n*̩�A�*

epsilon��g��VS.       ��W�	�u�*̩�A�* 

Average reward per step��g���ٳ       ��2	�v�*̩�A�*

epsilon��g�O�*1.       ��W�	��*̩�A�* 

Average reward per step��g�2 |       ��2	؜�*̩�A�*

epsilon��g�NN��.       ��W�	�<�*̩�A�* 

Average reward per step��g�ړ�       ��2	>�*̩�A�*

epsilon��g����.       ��W�	o�*̩�A�* 

Average reward per step��g���       ��2	^�*̩�A�*

epsilon��g��Cp�.       ��W�	���*̩�A�* 

Average reward per step��g���z;       ��2	x��*̩�A�*

epsilon��g�����.       ��W�	fi�*̩�A�* 

Average reward per step��g����       ��2	Mj�*̩�A�*

epsilon��g���R0       ���_	z��*̩�AY*#
!
Average reward per episodet����SeK.       ��W�	3��*̩�AY*!

total reward per episode  �2�_.       ��W�	�'�*̩�A�* 

Average reward per stept���x�qz       ��2	�(�*̩�A�*

epsilont���6��.       ��W�	M��*̩�A�* 

Average reward per stept����V��       ��2	w��*̩�A�*

epsilont�����9.       ��W�	aQ�*̩�A�* 

Average reward per stept�������       ��2	vR�*̩�A�*

epsilont����͍�.       ��W�	}" +̩�A�* 

Average reward per stept���w��e       ��2	F# +̩�A�*

epsilont�����O.       ��W�	��C+̩�A�* 

Average reward per stept���{/�I       ��2	d�C+̩�A�*

epsilont�����r.       ��W�	W�F+̩�A�* 

Average reward per stept���
�r!       ��2	|�F+̩�A�*

epsilont�����˘.       ��W�	t~H+̩�A�* 

Average reward per stept�����p       ��2	JH+̩�A�*

epsilont����l�O.       ��W�	�jJ+̩�A�* 

Average reward per stept���/Y=       ��2	�kJ+̩�A�*

epsilont����VB.       ��W�	��L+̩�A�* 

Average reward per stept���x�       ��2	��L+̩�A�*

epsilont�����T.       ��W�	�"O+̩�A�* 

Average reward per stept����9�       ��2	1$O+̩�A�*

epsilont���	`E�.       ��W�	�R+̩�A�* 

Average reward per stept����}��       ��2	 �R+̩�A�*

epsilont���.,)=.       ��W�	�7U+̩�A�* 

Average reward per stept���4̴�       ��2	�8U+̩�A�*

epsilont���n1*�.       ��W�	�+Y+̩�A�* 

Average reward per stept����IK�       ��2	�,Y+̩�A�*

epsilont�����,.       ��W�	2\+̩�A�* 

Average reward per stept����ŔU       ��2	\+̩�A�*

epsilont����1�.       ��W�	�_+̩�A�* 

Average reward per stept���a�T}       ��2	/�_+̩�A�*

epsilont���y�i.       ��W�	�wc+̩�A�* 

Average reward per stept����Q�       ��2	yc+̩�A�*

epsilont������.       ��W�	t@e+̩�A�* 

Average reward per stept����J+�       ��2	�Be+̩�A�*

epsilont������.       ��W�	/�h+̩�A�* 

Average reward per stept���	5T�       ��2	�h+̩�A�*

epsilont�������.       ��W�	QOl+̩�A�* 

Average reward per stept���29�       ��2	'Pl+̩�A�*

epsilont���Ũ�z.       ��W�	q�o+̩�A�* 

Average reward per stept���m�"       ��2	��o+̩�A�*

epsilont������.       ��W�	7�q+̩�A�* 

Average reward per stept�����^       ��2	e�q+̩�A�*

epsilont������.       ��W�	�:t+̩�A�* 

Average reward per stept������k       ��2	�;t+̩�A�*

epsilont���ԟ��.       ��W�	�	x+̩�A�* 

Average reward per stept�����D�       ��2	�
x+̩�A�*

epsilont�����*�.       ��W�	�hz+̩�A�* 

Average reward per stept���j��       ��2	�iz+̩�A�*

epsilont���-�D�.       ��W�	�>~+̩�A�* 

Average reward per stept���􊩓       ��2	�?~+̩�A�*

epsilont���ߦ��0       ���_	�e~+̩�AZ*#
!
Average reward per episode����3���.       ��W�	�f~+̩�AZ*!

total reward per episode   �z��&.       ��W�	�U�+̩�A�* 

Average reward per step����	7��       ��2	�V�+̩�A�*

epsilon�����X�.       ��W�	�Ɉ+̩�A�* 

Average reward per step����e�Xw       ��2	`ʈ+̩�A�*

epsilon����LGD�.       ��W�	2�+̩�A�* 

Average reward per step����3�C)       ��2	��+̩�A�*

epsilon�������!.       ��W�	Y��+̩�A�* 

Average reward per step����W��       ��2	D��+̩�A�*

epsilon����*q�Y.       ��W�	���+̩�A�* 

Average reward per step������L       ��2	���+̩�A�*

epsilon����ME�.       ��W�	�3�+̩�A�* 

Average reward per step�����7U       ��2	n4�+̩�A�*

epsilon����|NU�.       ��W�	�n�+̩�A�* 

Average reward per step�����t�       ��2	�o�+̩�A�*

epsilon����9��5.       ��W�	�!�+̩�A�* 

Average reward per step������
       ��2	�"�+̩�A�*

epsilon����G�N�.       ��W�	�+̩�A�* 

Average reward per step������       ��2	��+̩�A�*

epsilon����	@�F.       ��W�	���+̩�A�* 

Average reward per step����� �       ��2	ǝ�+̩�A�*

epsilon����U	.       ��W�	��+̩�A�* 

Average reward per step�����L�       ��2	��+̩�A�*

epsilon�������C.       ��W�	鹤+̩�A�* 

Average reward per step����j}+Z       ��2	
��+̩�A�*

epsilon�����C�C.       ��W�	,G�+̩�A�* 

Average reward per step�����d��       ��2	,H�+̩�A�*

epsilon����&�U�.       ��W�	*��+̩�A�* 

Average reward per step�����"�       ��2	��+̩�A�*

epsilon����[�V�.       ��W�	���+̩�A�* 

Average reward per step���� 7�^       ��2	���+̩�A�*

epsilon�����-��.       ��W�	y��+̩�A�* 

Average reward per step����}O��       ��2	���+̩�A�*

epsilon����fa�I.       ��W�	�X�+̩�A�* 

Average reward per step�����<�       ��2	\Y�+̩�A�*

epsilon�����W.       ��W�	��+̩�A�* 

Average reward per step�����b�)       ��2	��+̩�A�*

epsilon������.       ��W�	9�+̩�A�* 

Average reward per step��������       ��2	�+̩�A�*

epsilon�������\.       ��W�	%�+̩�A�* 

Average reward per step����l]p       ��2	�+̩�A�*

epsilon�����6�V.       ��W�	ޭ�+̩�A�* 

Average reward per step�����9�       ��2	���+̩�A�*

epsilon����#o� .       ��W�	��+̩�A�* 

Average reward per step����Ġi       ��2	��+̩�A�*

epsilon�����\��.       ��W�	B��+̩�A�* 

Average reward per step����2Ur       ��2	:��+̩�A�*

epsilon������.       ��W�	1(�+̩�A�* 

Average reward per step�������       ��2	=)�+̩�A�*

epsilon����� Q�.       ��W�	D��+̩�A�* 

Average reward per step�����v�       ��2	j��+̩�A�*

epsilon�����r�0       ���_	g,�+̩�A[*#
!
Average reward per episode�Q��\�W
.       ��W�	8-�+̩�A[*!

total reward per episode  �N슟.       ��W�	�0�+̩�A�* 

Average reward per step�Q��]��=       ��2	�1�+̩�A�*

epsilon�Q���k�[.       ��W�	u��+̩�A�* 

Average reward per step�Q��+Oֈ       ��2	>��+̩�A�*

epsilon�Q��6�u.       ��W�	���+̩�A�* 

Average reward per step�Q��G�`w       ��2	���+̩�A�*

epsilon�Q���G�.       ��W�	��+̩�A�* 

Average reward per step�Q��dh�       ��2	5��+̩�A�*

epsilon�Q���g�.       ��W�	�p�+̩�A�* 

Average reward per step�Q���5�       ��2	&r�+̩�A�*

epsilon�Q��9��Q.       ��W�	�U�+̩�A�* 

Average reward per step�Q����'       ��2	�V�+̩�A�*

epsilon�Q��e���.       ��W�	���+̩�A�* 

Average reward per step�Q���2c       ��2	���+̩�A�*

epsilon�Q���5U�.       ��W�	H��+̩�A�* 

Average reward per step�Q���.�~       ��2	a��+̩�A�*

epsilon�Q���j��.       ��W�	�#�+̩�A�* 

Average reward per step�Q��[,�       ��2	�$�+̩�A�*

epsilon�Q���9�.       ��W�	O��+̩�A�* 

Average reward per step�Q��87�       ��2	���+̩�A�*

epsilon�Q���1lV.       ��W�	���+̩�A�* 

Average reward per step�Q��2�E       ��2	���+̩�A�*

epsilon�Q��` �.       ��W�	���+̩�A�* 

Average reward per step�Q��|�[       ��2	���+̩�A�*

epsilon�Q�����.       ��W�	a�+̩�A�* 

Average reward per step�Q�����       ��2	m�+̩�A�*

epsilon�Q��s�	=.       ��W�	n��+̩�A�* 

Average reward per step�Q��R�b       ��2	/��+̩�A�*

epsilon�Q��t�nh.       ��W�	��+̩�A�* 

Average reward per step�Q���       ��2	��+̩�A�*

epsilon�Q���$�.       ��W�	Ŏ�+̩�A�* 

Average reward per step�Q����!�       ��2	?��+̩�A�*

epsilon�Q���!.       ��W�	ak�+̩�A�* 

Average reward per step�Q��R%Я       ��2	l�+̩�A�*

epsilon�Q���M>�.       ��W�	=��+̩�A�* 

Average reward per step�Q���F�\       ��2	��+̩�A�*

epsilon�Q�����.       ��W�	!��+̩�A�* 

Average reward per step�Q����)w       ��2	 ��+̩�A�*

epsilon�Q��ؚY�.       ��W�	v4�+̩�A�* 

Average reward per step�Q���k��       ��2	]5�+̩�A�*

epsilon�Q�����.       ��W�	d�,̩�A�* 

Average reward per step�Q��һ�v       ��2	�,̩�A�*

epsilon�Q��:I��.       ��W�	�],̩�A�* 

Average reward per step�Q��a��l       ��2	x^,̩�A�*

epsilon�Q���N��.       ��W�	��	,̩�A�* 

Average reward per step�Q��B�:       ��2	��	,̩�A�*

epsilon�Q��H�w.       ��W�	�:,̩�A�* 

Average reward per step�Q��޻I       ��2	�;,̩�A�*

epsilon�Q���۷.       ��W�	,̩�A�* 

Average reward per step�Q���G       ��2	n,̩�A�*

epsilon�Q��^b.       ��W�	�,̩�A�* 

Average reward per step�Q�����       ��2	�,̩�A�*

epsilon�Q��[�W.       ��W�	߇,̩�A�* 

Average reward per step�Q��ME��       ��2		�,̩�A�*

epsilon�Q��c�2.       ��W�	`�,̩�A�* 

Average reward per step�Q���*�       ��2	B�,̩�A�*

epsilon�Q��"D�.       ��W�	�w,̩�A�* 

Average reward per step�Q��{��a       ��2	�x,̩�A�*

epsilon�Q��5��w.       ��W�	V�,̩�A�* 

Average reward per step�Q����n2       ��2	�,̩�A�*

epsilon�Q���&M\.       ��W�	� ,̩�A�* 

Average reward per step�Q�����       ��2	"� ,̩�A�*

epsilon�Q��]�rD.       ��W�	E%,̩�A�* 

Average reward per step�Q���s�H       ��2	�%,̩�A�*

epsilon�Q��[3�8.       ��W�	&�(,̩�A�* 

Average reward per step�Q��W.�       ��2	3�(,̩�A�*

epsilon�Q��;�=.       ��W�	)v*,̩�A�* 

Average reward per step�Q���z�r       ��2	}w*,̩�A�*

epsilon�Q���A.       ��W�	�,,̩�A�* 

Average reward per step�Q���u�       ��2	9�,,̩�A�*

epsilon�Q������.       ��W�	P/,̩�A�* 

Average reward per step�Q�����       ��2	iQ/,̩�A�*

epsilon�Q��z��.       ��W�	�x2,̩�A�* 

Average reward per step�Q�����&       ��2	}y2,̩�A�*

epsilon�Q��J�.       ��W�	�4,̩�A�* 

Average reward per step�Q��z6k:       ��2	5�4,̩�A�*

epsilon�Q�����.       ��W�	�>7,̩�A�* 

Average reward per step�Q��E�c       ��2	�?7,̩�A�*

epsilon�Q�����.       ��W�	�:,̩�A�* 

Average reward per step�Q���ɵ       ��2	9�:,̩�A�*

epsilon�Q��y�F.       ��W�	Y�=,̩�A�* 

Average reward per step�Q����u,       ��2	e�=,̩�A�*

epsilon�Q���.       ��W�	1?�,̩�A�* 

Average reward per step�Q��?sO       ��2	h@�,̩�A�*

epsilon�Q����.       ��W�	��,̩�A�* 

Average reward per step�Q�����       ��2	��,̩�A�*

epsilon�Q��4E_�.       ��W�	���,̩�A�* 

Average reward per step�Q����[�       ��2	���,̩�A�*

epsilon�Q��l}�.       ��W�	���,̩�A�* 

Average reward per step�Q��jsҦ       ��2	կ�,̩�A�*

epsilon�Q��><q.       ��W�	���,̩�A�* 

Average reward per step�Q�����i       ��2	���,̩�A�*

epsilon�Q��}]�2.       ��W�	1_�,̩�A�* 

Average reward per step�Q��3��       ��2	c`�,̩�A�*

epsilon�Q��d��.       ��W�	���,̩�A�* 

Average reward per step�Q����L�       ��2	���,̩�A�*

epsilon�Q����ۆ.       ��W�	Ț�,̩�A�* 

Average reward per step�Q��8�1�       ��2	Ǜ�,̩�A�*

epsilon�Q��Y�d.       ��W�	j��,̩�A�* 

Average reward per step�Q�����/       ��2	e��,̩�A�*

epsilon�Q��|ȃ�.       ��W�	DL�,̩�A�* 

Average reward per step�Q���\�       ��2	@M�,̩�A�*

epsilon�Q���7�q.       ��W�	���,̩�A�* 

Average reward per step�Q��:K�#       ��2	���,̩�A�*

epsilon�Q���.�~.       ��W�	`Z�,̩�A�* 

Average reward per step�Q��"�_�       ��2	6[�,̩�A�*

epsilon�Q��x�FT.       ��W�	ɫ�,̩�A�* 

Average reward per step�Q��	�|�       ��2	���,̩�A�*

epsilon�Q��"Ͳ�.       ��W�	�A�,̩�A�* 

Average reward per step�Q��(�,�       ��2	�B�,̩�A�*

epsilon�Q��z�o�.       ��W�	���,̩�A�* 

Average reward per step�Q���P<�       ��2	���,̩�A�*

epsilon�Q��KA=�.       ��W�	}��,̩�A�* 

Average reward per step�Q����j       ��2	S��,̩�A�*

epsilon�Q��_|1$.       ��W�	�8�,̩�A�* 

Average reward per step�Q����%�       ��2	K9�,̩�A�*

epsilon�Q��W�1�0       ���_	GU�,̩�A\*#
!
Average reward per episodej�����z�.       ��W�	�U�,̩�A\*!

total reward per episode  ��r�M�.       ��W�	-#�,̩�A�* 

Average reward per stepj�����       ��2	�#�,̩�A�*

epsilonj���fKh�.       ��W�	l��,̩�A�* 

Average reward per stepj���v��       ��2	t��,̩�A�*

epsilonj����%.       ��W�	��,̩�A�* 

Average reward per stepj���a��M       ��2	y �,̩�A�*

epsilonj����Ku,.       ��W�	��,̩�A�* 

Average reward per stepj���3���       ��2	���,̩�A�*

epsilonj���e7�p.       ��W�	��,̩�A�* 

Average reward per stepj�����)       ��2	K�,̩�A�*

epsilonj���M�ƥ.       ��W�	�S-̩�A�* 

Average reward per stepj����!�       ��2	�T-̩�A�*

epsilonj���݂�l.       ��W�	X�-̩�A�* 

Average reward per stepj���g�i�       ��2	�-̩�A�*

epsilonj������z.       ��W�	�-̩�A�* 

Average reward per stepj����BJ�       ��2	��-̩�A�*

epsilonj���Q��W.       ��W�	Q1-̩�A�* 

Average reward per stepj����)0C       ��2	32-̩�A�*

epsilonj����8��.       ��W�	��	-̩�A�* 

Average reward per stepj������I       ��2	��	-̩�A�*

epsilonj�����.V.       ��W�	(-̩�A�* 

Average reward per stepj�����H�       ��2	�(-̩�A�*

epsilonj������.       ��W�	Ҭ-̩�A�* 

Average reward per stepj���!"p
       ��2	�-̩�A�*

epsilonj���~.       ��W�	U-̩�A�* 

Average reward per stepj���bcz       ��2	PV-̩�A�*

epsilonj����e��.       ��W�	��-̩�A�* 

Average reward per stepj���KH�       ��2	Á-̩�A�*

epsilonj���q �.       ��W�	��-̩�A�* 

Average reward per stepj������~       ��2	��-̩�A�*

epsilonj���W�A.       ��W�	�-̩�A�* 

Average reward per stepj����~V       ��2	�-̩�A�*

epsilonj���!	N�0       ���_	��-̩�A]*#
!
Average reward per episode  �'CKf.       ��W�	��-̩�A]*!

total reward per episode  �D.3F.       ��W�	%�-̩�A�* 

Average reward per step  �%� �       ��2	��-̩�A�*

epsilon  �FZV.       ��W�	�%-̩�A�* 

Average reward per step  �l:       ��2	�&-̩�A�*

epsilon  �-�f�.       ��W�	K�-̩�A�* 

Average reward per step  �����       ��2	:�-̩�A�*

epsilon  �� z.       ��W�	I�!-̩�A�* 

Average reward per step  ��@u�       ��2	8�!-̩�A�*

epsilon  ��i{�.       ��W�	_$-̩�A�* 

Average reward per step  �7�       ��2	-$-̩�A�*

epsilon  ���!�.       ��W�	�s&-̩�A�* 

Average reward per step  �i��       ��2	�t&-̩�A�*

epsilon  ���lR.       ��W�	;(-̩�A�* 

Average reward per step  �v�q=       ��2	z(-̩�A�*

epsilon  ��".       ��W�	\*-̩�A�* 

Average reward per step  �k�r�       ��2	]*-̩�A�*

epsilon  �I�.       ��W�	u�,-̩�A�* 

Average reward per step  �Q�VI       ��2	-�,-̩�A�*

epsilon  �c��.       ��W�	=�.-̩�A�* 

Average reward per step  �L��x       ��2	J�.-̩�A�*

epsilon  �y��:.       ��W�	Re3-̩�A�* 

Average reward per step  ���φ       ��2	Ef3-̩�A�*

epsilon  �y��r.       ��W�	b7-̩�A�* 

Average reward per step  �=�K       ��2	M7-̩�A�*

epsilon  ��[gl.       ��W�	��8-̩�A�* 

Average reward per step  �J�       ��2	/�8-̩�A�*

epsilon  �΃�.       ��W�	�;-̩�A�* 

Average reward per step  �<�b)       ��2	o;-̩�A�*

epsilon  �Έ�>.       ��W�	��<-̩�A�* 

Average reward per step  ���Wp       ��2	`�<-̩�A�*

epsilon  ��oXe.       ��W�	o�>-̩�A�* 

Average reward per step  ���AF       ��2	Z�>-̩�A�*

epsilon  �^Q�.       ��W�	��@-̩�A�* 

Average reward per step  ��hv`       ��2	��@-̩�A�*

epsilon  �#w�).       ��W�	sC-̩�A�* 

Average reward per step  �C�?�       ��2	�C-̩�A�*

epsilon  ��Š.       ��W�	ŬD-̩�A�* 

Average reward per step  ��v)       ��2	��D-̩�A�*

epsilon  ����.       ��W�	K�G-̩�A�* 

Average reward per step  ���*�       ��2	K�G-̩�A�*

epsilon  �X�Ș.       ��W�	��K-̩�A�* 

Average reward per step  � ��       ��2	ρK-̩�A�*

epsilon  ����l.       ��W�	�M-̩�A�* 

Average reward per step  �w �d       ��2	hM-̩�A�*

epsilon  ���n�.       ��W�	^fO-̩�A�* 

Average reward per step  �v)�o       ��2	�gO-̩�A�*

epsilon  ��/x{.       ��W�	9�Q-̩�A�* 

Average reward per step  ���       ��2	�Q-̩�A�*

epsilon  �^�.       ��W�	z5S-̩�A�* 

Average reward per step  �`dθ       ��2	?6S-̩�A�*

epsilon  ����.       ��W�	�U-̩�A�* 

Average reward per step  �#�]6       ��2	��U-̩�A�*

epsilon  �Ep0       ���_	��U-̩�A^*#
!
Average reward per episode���Y�.       ��W�	��U-̩�A^*!

total reward per episode  �`'�.       ��W�	L�Y-̩�A�* 

Average reward per step����Z�x       ��2	X�Y-̩�A�*

epsilon���j�ES.       ��W�	V�[-̩�A�* 

Average reward per step�����N�       ��2	
�[-̩�A�*

epsilon�����.       ��W�	H4^-̩�A�* 

Average reward per step����t�       ��2	5^-̩�A�*

epsilon���sz.       ��W�	�y`-̩�A�* 

Average reward per step���8���       ��2	�z`-̩�A�*

epsilon������j.       ��W�	�hd-̩�A�* 

Average reward per step���w�R_       ��2	�id-̩�A�*

epsilon���/c��.       ��W�	�4f-̩�A�* 

Average reward per step���)Ys�       ��2	�5f-̩�A�*

epsilon����;�.       ��W�	nmh-̩�A�* 

Average reward per step������       ��2	nh-̩�A�*

epsilon���6���.       ��W�	ܼj-̩�A�* 

Average reward per step����t�       ��2	��j-̩�A�*

epsilon���M��.       ��W�	:Zn-̩�A�* 

Average reward per step���j@��       ��2	%[n-̩�A�*

epsilon���{5�.       ��W�	$�p-̩�A�* 

Average reward per step���y��       ��2	�p-̩�A�*

epsilon���Z�,.       ��W�	rNr-̩�A�* 

Average reward per step���u]�       ��2	jOr-̩�A�*

epsilon�����.       ��W�	��t-̩�A�* 

Average reward per step����B��       ��2	J�t-̩�A�*

epsilon���#x�.       ��W�	�v-̩�A�* 

Average reward per step����5-       ��2	,�v-̩�A�*

epsilon���Yw�O.       ��W�	
�x-̩�A�* 

Average reward per step����V�O       ��2	=�x-̩�A�*

epsilon���C?�|.       ��W�	�Wz-̩�A�* 

Average reward per step��� �.�       ��2	�Xz-̩�A�*

epsilon������5.       ��W�	mp}-̩�A�* 

Average reward per step�����+       ��2	;q}-̩�A�*

epsilon���*�e�.       ��W�	�!�-̩�A�* 

Average reward per step���:<P�       ��2	�"�-̩�A�*

epsilon���
^�.       ��W�	���-̩�A�* 

Average reward per step����� �       ��2	��-̩�A�*

epsilon���F��.       ��W�	�م-̩�A�* 

Average reward per step���B�       ��2	fڅ-̩�A�*

epsilon���Y�b�.       ��W�	 }�-̩�A�* 

Average reward per step����]�       ��2	�}�-̩�A�*

epsilon������v.       ��W�	U�-̩�A�* 

Average reward per step���l�(       ��2	�U�-̩�A�*

epsilon���-�)K.       ��W�	���-̩�A�* 

Average reward per step���G{       ��2	���-̩�A�*

epsilon���_X��.       ��W�	A��-̩�A�* 

Average reward per step����4�       ��2	=��-̩�A�*

epsilon���|��G.       ��W�	�k�-̩�A�* 

Average reward per step�����N       ��2	nl�-̩�A�*

epsilon�����.       ��W�	Ƥ�-̩�A�* 

Average reward per step����즫       ��2	襕-̩�A�*

epsilon�������.       ��W�	.ɗ-̩�A�* 

Average reward per step����w�       ��2	�ɗ-̩�A�*

epsilon������3.       ��W�	f��-̩�A�* 

Average reward per step���ȱ�       ��2	]��-̩�A�*

epsilon�����1.       ��W�	��-̩�A�* 

Average reward per step���G��I       ��2	J�-̩�A�*

epsilon���d��.       ��W�	�-̩�A�* 

Average reward per step���3� +       ��2	��-̩�A�*

epsilon����;�.       ��W�	=)�-̩�A�* 

Average reward per step���.0<7       ��2	*�-̩�A�*

epsilon���Бyn.       ��W�	�a�-̩�A�* 

Average reward per step����$y       ��2	�b�-̩�A�*

epsilon����{��.       ��W�	��-̩�A�* 

Average reward per step���:o�       ��2	��-̩�A�*

epsilon���k(�.       ��W�	���-̩�A�* 

Average reward per step���A���       ��2	���-̩�A�*

epsilon����j�.       ��W�	sd�-̩�A�* 

Average reward per step���q�
       ��2	^e�-̩�A�*

epsilon����\�.       ��W�	Ė�-̩�A�* 

Average reward per step���j̥�       ��2	���-̩�A�*

epsilon����@.       ��W�	NF�-̩�A�* 

Average reward per step���Jb�^       ��2	AG�-̩�A�*

epsilon����;�F.       ��W�	К�-̩�A�* 

Average reward per step���F6�       ��2	���-̩�A�*

epsilon���z��.       ��W�	�.�-̩�A�* 

Average reward per step����w�a       ��2	�/�-̩�A�*

epsilon���`���.       ��W�	��-̩�A�* 

Average reward per step���]Bè       ��2	��-̩�A�*

epsilon����Nb.       ��W�	e��-̩�A�* 

Average reward per step�����غ       ��2	z��-̩�A�*

epsilon������0.       ��W�	8��-̩�A�* 

Average reward per step���f2,�       ��2	���-̩�A�*

epsilon�����).       ��W�	,�-̩�A�* 

Average reward per step���.EJ       ��2	�,�-̩�A�*

epsilon���a�`.       ��W�	�c�-̩�A�* 

Average reward per step���Ȕy�       ��2	,e�-̩�A�*

epsilon�����rE.       ��W�	��-̩�A�* 

Average reward per step�����L       ��2	T��-̩�A�*

epsilon�����w�.       ��W�	�'�-̩�A�* 

Average reward per step����)�W       ��2	c(�-̩�A�*

epsilon���.x�{.       ��W�	�{�-̩�A�* 

Average reward per step���c�e       ��2	W|�-̩�A�*

epsilon���\��7.       ��W�	*6�-̩�A�* 

Average reward per step���D��       ��2	�6�-̩�A�*

epsilon���8υ.       ��W�	j�-̩�A�* 

Average reward per step������#       ��2	��-̩�A�*

epsilon����p	�.       ��W�	ߤ�-̩�A�* 

Average reward per step����H�        ��2	���-̩�A�*

epsilon���+�ML.       ��W�	8H�-̩�A�* 

Average reward per step���o0�       ��2	�H�-̩�A�*

epsilon���5��.       ��W�	���-̩�A�* 

Average reward per step��� �˲       ��2	���-̩�A�*

epsilon��� $�.       ��W�	���-̩�A�* 

Average reward per step���Tj4�       ��2	���-̩�A�*

epsilon���o�.       ��W�	ݔ�-̩�A�* 

Average reward per step���y�       ��2	)��-̩�A�*

epsilon������k.       ��W�	�Q�-̩�A�* 

Average reward per step���-�ֱ       ��2	�R�-̩�A�*

epsilon���w9��.       ��W�	��-̩�A�* 

Average reward per step���n|3       ��2	���-̩�A�*

epsilon���Y�#�.       ��W�	d �-̩�A�* 

Average reward per step���I0@r       ��2	W!�-̩�A�*

epsilon�����D�.       ��W�	Ll�-̩�A�* 

Average reward per step���d/��       ��2	�l�-̩�A�*

epsilon����۴�.       ��W�	�3�-̩�A�* 

Average reward per step���rF�       ��2	�4�-̩�A�*

epsilon����(��.       ��W�	���-̩�A�* 

Average reward per step�����f       ��2	j��-̩�A�*

epsilon����e��.       ��W�	Z��-̩�A�* 

Average reward per step���|��a       ��2	��-̩�A�*

epsilon�����$�.       ��W�	<�-̩�A�* 

Average reward per step���/���       ��2	=�-̩�A�*

epsilon���6D��.       ��W�	��-̩�A�* 

Average reward per step���W�       ��2	��-̩�A�*

epsilon�����Xb.       ��W�	'J�-̩�A�* 

Average reward per step����.��       ��2	K�-̩�A�*

epsilon�����d.       ��W�	=��-̩�A�* 

Average reward per step���ŝ%�       ��2	��-̩�A�*

epsilon��� Z.       ��W�	Ő�-̩�A�* 

Average reward per step������       ��2	���-̩�A�*

epsilon�����.       ��W�	z� .̩�A�* 

Average reward per step�������       ��2	v� .̩�A�*

epsilon�����,�.       ��W�	/.̩�A�* 

Average reward per step���� v1       ��2	.̩�A�*

epsilon���n�E.       ��W�	�Q.̩�A�* 

Average reward per step�����0       ��2	eR.̩�A�*

epsilon����6�0       ���_	zl.̩�A_*#
!
Average reward per episodeZZھuo�H.       ��W�		m.̩�A_*!

total reward per episode  ��<0W�.       ��W�	�l	.̩�A�* 

Average reward per stepZZھ���       ��2	�m	.̩�A�*

epsilonZZھ��D/.       ��W�	�X.̩�A�* 

Average reward per stepZZھ���       ��2	�Y.̩�A�*

epsilonZZھ��]�.       ��W�	W�.̩�A�* 

Average reward per stepZZھҫ       ��2	$�.̩�A�*

epsilonZZھh��.       ��W�	&�.̩�A�* 

Average reward per stepZZھo�,       ��2	�.̩�A�*

epsilonZZھJR.       ��W�	��.̩�A�* 

Average reward per stepZZھ[�/�       ��2	��.̩�A�*

epsilonZZھ���}.       ��W�	�'.̩�A�* 

Average reward per stepZZھ�pZ       ��2	,).̩�A�*

epsilonZZھ}��U.       ��W�	��.̩�A�* 

Average reward per stepZZھ�,��       ��2	��.̩�A�*

epsilonZZھ���.       ��W�		3.̩�A�* 

Average reward per stepZZھA(       ��2	4.̩�A�*

epsilonZZھ�Ys�.       ��W�	�_.̩�A�* 

Average reward per stepZZھZ�C�       ��2	�`.̩�A�*

epsilonZZھ�=� .       ��W�	�.̩�A�* 

Average reward per stepZZھ��v       ��2	�.̩�A�*

epsilonZZھǜ��.       ��W�	�/".̩�A�* 

Average reward per stepZZھ�i�       ��2	Y0".̩�A�*

epsilonZZھ-�D.       ��W�	3�#.̩�A�* 

Average reward per stepZZھ9Z�[       ��2	��#.̩�A�*

epsilonZZھ��iK.       ��W�	<&.̩�A�* 

Average reward per stepZZھ�hE�       ��2	�<&.̩�A�*

epsilonZZھ�X��.       ��W�		l(.̩�A�* 

Average reward per stepZZھs�;�       ��2		m(.̩�A�*

epsilonZZھ�ʠ.       ��W�	AG*.̩�A�* 

Average reward per stepZZھ�<��       ��2	
H*.̩�A�*

epsilonZZھ�yS�.       ��W�	��+.̩�A�* 

Average reward per stepZZھjZ��       ��2	�+.̩�A�*

epsilonZZھ�qY .       ��W�	��..̩�A�* 

Average reward per stepZZھ]�û       ��2	O�..̩�A�*

epsilonZZھ�~P.       ��W�	�1.̩�A�* 

Average reward per stepZZھ���:       ��2	�1.̩�A�*

epsilonZZھJ�.       ��W�	u95.̩�A�* 

Average reward per stepZZھ&4��       ��2	\:5.̩�A�*

epsilonZZھ�Lh�0       ���_	�r5.̩�A`*#
!
Average reward per episodey�]~ݰ.       ��W�	�s5.̩�A`*!

total reward per episode  �OE�.       ��W�	�D;.̩�A�* 

Average reward per stepy�ɝ�       ��2	�E;.̩�A�*

epsilony���B}.       ��W�	s�>.̩�A�* 

Average reward per stepy��h�       ��2	f�>.̩�A�*

epsilony���+.       ��W�	A.̩�A�* 

Average reward per stepy�d��       ��2	�A.̩�A�*

epsilony��.       ��W�	�B.̩�A�* 

Average reward per stepy�^�v5       ��2	��B.̩�A�*

epsilony�\h�.       ��W�	�@E.̩�A�* 

Average reward per stepy�֝R�       ��2	�AE.̩�A�*

epsilony�//:l.       ��W�	��G.̩�A�* 

Average reward per stepy���n#       ��2	n�G.̩�A�*

epsilony���{�.       ��W�	�;K.̩�A�* 

Average reward per stepy�d�n�       ��2	�<K.̩�A�*

epsilony�,��.       ��W�	�M.̩�A�* 

Average reward per stepy�IN�{       ��2	�M.̩�A�*

epsilony��i��.       ��W�	��O.̩�A�* 

Average reward per stepy��!�       ��2	@�O.̩�A�*

epsilony�IU�.       ��W�	Rѓ.̩�A�* 

Average reward per stepy��Ǳ       ��2	Jғ.̩�A�*

epsilony����z.       ��W�	˂�.̩�A�* 

Average reward per stepy�z|B`       ��2	{��.̩�A�*

epsilony�Ri�".       ��W�	���.̩�A�* 

Average reward per stepy��;�E       ��2	¡�.̩�A�*

epsilony�����.       ��W�	�˙.̩�A�* 

Average reward per stepy���       ��2	�̙.̩�A�*

epsilony���.       ��W�	�v�.̩�A�* 

Average reward per stepy�pYy       ��2	yw�.̩�A�*

epsilony� �.       ��W�	�ǝ.̩�A�* 

Average reward per stepy��{�e       ��2	�ȝ.̩�A�*

epsilony�wI6t.       ��W�	WA�.̩�A�* 

Average reward per stepy�����       ��2	5B�.̩�A�*

epsilony�f'�O.       ��W�	���.̩�A�* 

Average reward per stepy���P       ��2	��.̩�A�*

epsilony��Q.       ��W�	���.̩�A�* 

Average reward per stepy��iԿ       ��2	n��.̩�A�*

epsilony��u`C.       ��W�	���.̩�A�* 

Average reward per stepy���Z       ��2	���.̩�A�*

epsilony�שf.       ��W�	��.̩�A�* 

Average reward per stepy���ò       ��2	��.̩�A�*

epsilony�C�K1.       ��W�	���.̩�A�* 

Average reward per stepy��y       ��2	a��.̩�A�*

epsilony���1.       ��W�	l^�.̩�A�* 

Average reward per stepy���G�       ��2	J_�.̩�A�*

epsilony�����.       ��W�	��.̩�A�* 

Average reward per stepy�@C��       ��2	ɰ�.̩�A�*

epsilony�4#��.       ��W�	'.�.̩�A�* 

Average reward per stepy��
�n       ��2	�.�.̩�A�*

epsilony�H���.       ��W�	{��.̩�A�* 

Average reward per stepy����f       ��2	n��.̩�A�*

epsilony�#Wƀ.       ��W�	q�.̩�A�* 

Average reward per stepy�ЀIX       ��2	u�.̩�A�*

epsilony��>�.       ��W�	Nc�.̩�A�* 

Average reward per stepy��꓊       ��2	0d�.̩�A�*

epsilony�LE�_.       ��W�	���.̩�A�* 

Average reward per stepy�A��a       ��2	q��.̩�A�*

epsilony���o�.       ��W�	�
�.̩�A�* 

Average reward per stepy�����       ��2	��.̩�A�*

epsilony�H�b0       ���_	f.�.̩�Aa*#
!
Average reward per episode���Q.       ��W�	8/�.̩�Aa*!

total reward per episode  �[�	�.       ��W�	:=/̩�A�* 

Average reward per step���3Q�O       ��2	>/̩�A�*

epsilon��� 6.       ��W�	�:/̩�A�* 

Average reward per step�����Y       ��2	�;/̩�A�*

epsilon���J�.       ��W�	��/̩�A�* 

Average reward per step����M�       ��2	K�/̩�A�*

epsilon����1p�.       ��W�	'g/̩�A�* 

Average reward per step�����>       ��2	h/̩�A�*

epsilon���ޅ� .       ��W�	��/̩�A�* 

Average reward per step����v`�       ��2	`�/̩�A�*

epsilon���\���.       ��W�	�iR/̩�A�* 

Average reward per step�����?�       ��2		kR/̩�A�*

epsilon���\:r�.       ��W�	|cU/̩�A�* 

Average reward per step���x�"       ��2	dU/̩�A�*

epsilon���!�B�.       ��W�	�|W/̩�A�* 

Average reward per step���Cz�s       ��2	A}W/̩�A�*

epsilon���
/b�.       ��W�	�AZ/̩�A�* 

Average reward per step����Ӄ�       ��2	�BZ/̩�A�*

epsilon���}.       ��W�	e�\/̩�A�* 

Average reward per step����ZP�       ��2	�\/̩�A�*

epsilon����K�B.       ��W�	 8`/̩�A�* 

Average reward per step�����+       ��2	�8`/̩�A�*

epsilon���kD�.       ��W�		�a/̩�A�* 

Average reward per step���r���       ��2	��a/̩�A�*

epsilon����wD%.       ��W�	h#d/̩�A�* 

Average reward per step���?n�       ��2	W$d/̩�A�*

epsilon������.       ��W�	��f/̩�A�* 

Average reward per step�����1       ��2	��f/̩�A�*

epsilon�����x.       ��W�	�h/̩�A�* 

Average reward per step������i       ��2	�h/̩�A�*

epsilon�����o�.       ��W�	�Tj/̩�A�* 

Average reward per step����T7p       ��2	�Uj/̩�A�*

epsilon�����#.       ��W�	ܜl/̩�A�* 

Average reward per step�����g       ��2	Нl/̩�A�*

epsilon���q��Q.       ��W�	an/̩�A�* 

Average reward per step�����       ��2	bn/̩�A�*

epsilon���W�_�.       ��W�	�p/̩�A�* 

Average reward per step���X�D`       ��2	�p/̩�A�*

epsilon���K騃.       ��W�	Tr/̩�A�* 

Average reward per step������       ��2	&Ur/̩�A�*

epsilon���Ή��.       ��W�	I�t/̩�A�* 

Average reward per step����}o       ��2	�t/̩�A�*

epsilon�����|.       ��W�	��v/̩�A�* 

Average reward per step������       ��2	z�v/̩�A�*

epsilon��� �.       ��W�	�3y/̩�A�* 

Average reward per step���E�I       ��2	/4y/̩�A�*

epsilon����0.       ��W�	p�|/̩�A�* 

Average reward per step���`�c�       ��2	�|/̩�A�*

epsilon���b�L\.       ��W�	�9/̩�A�* 

Average reward per step����t'�       ��2	�:/̩�A�*

epsilon������G.       ��W�	�ˀ/̩�A�* 

Average reward per step����\H�       ��2	�̀/̩�A�*

epsilon���ƃ�.       ��W�	�6�/̩�A�* 

Average reward per step����}�       ��2	�7�/̩�A�*

epsilon����Al.       ��W�	���/̩�A�* 

Average reward per step����qׯ       ��2	r/̩�A�*

epsilon���L��.       ��W�	d�/̩�A�* 

Average reward per step�����Z"       ��2	F�/̩�A�*

epsilon����jO.       ��W�	FB�/̩�A�* 

Average reward per step���-z       ��2	�B�/̩�A�*

epsilon������0       ���_	f��/̩�Ab*#
!
Average reward per episodeDD��ǂ��.       ��W�	���/̩�Ab*!

total reward per episode  �ow��.       ��W�	}Ѝ/̩�A�* 

Average reward per stepDD�����       ��2	Jэ/̩�A�*

epsilonDD��8�6.       ��W�	r��/̩�A�* 

Average reward per stepDD�����       ��2	���/̩�A�*

epsilonDD���'M�.       ��W�	��/̩�A�* 

Average reward per stepDD��p�j       ��2	��/̩�A�*

epsilonDD���%.       ��W�	��/̩�A�* 

Average reward per stepDD��t�       ��2	G��/̩�A�*

epsilonDD���[�.       ��W�	X�/̩�A�* 

Average reward per stepDD��=Wc�       ��2	m�/̩�A�*

epsilonDD��~��.       ��W�	?�/̩�A�* 

Average reward per stepDD���S'%       ��2	%@�/̩�A�*

epsilonDD������.       ��W�	 �/̩�A�* 

Average reward per stepDD���-Sa       ��2	��/̩�A�*

epsilonDD���!r).       ��W�	�@�/̩�A�* 

Average reward per stepDD��(W       ��2	�A�/̩�A�*

epsilonDD��^%1�.       ��W�	o��/̩�A�* 

Average reward per stepDD�����       ��2	E��/̩�A�*

epsilonDD��/�Y.       ��W�	H��/̩�A�* 

Average reward per stepDD�����       ��2	&��/̩�A�*

epsilonDD����.       ��W�	j4�/̩�A�* 

Average reward per stepDD��A���       ��2	D5�/̩�A�*

epsilonDD���*�4.       ��W�	]��/̩�A�* 

Average reward per stepDD��00��       ��2	���/̩�A�*

epsilonDD��]��.       ��W�	y\�/̩�A�* 

Average reward per stepDD�����       ��2	`�/̩�A�*

epsilonDD���㈗.       ��W�	D0�/̩�A�* 

Average reward per stepDD�� �M$       ��2	�0�/̩�A�*

epsilonDD����h.       ��W�	�X�/̩�A�* 

Average reward per stepDD��E#�       ��2	�Y�/̩�A�*

epsilonDD��g��.       ��W�	�̲/̩�A�* 

Average reward per stepDD���H�       ��2	�Ͳ/̩�A�*

epsilonDD��x�ox.       ��W�	J�/̩�A�* 

Average reward per stepDD����       ��2	9�/̩�A�*

epsilonDD���S.       ��W�	ђ�/̩�A�* 

Average reward per stepDD��_���       ��2	���/̩�A�*

epsilonDD���*J�.       ��W�	���/̩�A�* 

Average reward per stepDD��m���       ��2	���/̩�A�*

epsilonDD��2gW�.       ��W�	$ռ/̩�A�* 

Average reward per stepDD��j��       ��2	�ռ/̩�A�*

epsilonDD��p��.       ��W�	8L�/̩�A�* 

Average reward per stepDD����G       ��2	M�/̩�A�*

epsilonDD��г�.       ��W�	S�/̩�A�* 

Average reward per stepDD��0��       ��2	��/̩�A�*

epsilonDD���qF�0       ���_	�&�/̩�Ac*#
!
Average reward per episode������b�.       ��W�	x'�/̩�Ac*!

total reward per episode  '��ʼ�.       ��W�	?�/̩�A�* 

Average reward per step�����ފ�       ��2	��/̩�A�*

epsilon����>�'�.       ��W�	H�/̩�A�* 

Average reward per step�����*+       ��2	�H�/̩�A�*

epsilon��������.       ��W�	�m�/̩�A�* 

Average reward per step����/c�       ��2	�n�/̩�A�*

epsilon����OrH�.       ��W�	�K�/̩�A�* 

Average reward per step����
<�       ��2	�L�/̩�A�*

epsilon�����smO.       ��W�	�;�/̩�A�* 

Average reward per step����Y_�7       ��2	�<�/̩�A�*

epsilon�����Y��.       ��W�	o��/̩�A�* 

Average reward per step����E��       ��2	V��/̩�A�*

epsilon�����5c.       ��W�	I0�/̩�A�* 

Average reward per step����P�f       ��2	'1�/̩�A�*

epsilon������{8.       ��W�	X�/̩�A�* 

Average reward per step����}Rw�       ��2	Y�/̩�A�*

epsilon����t��.       ��W�	���/̩�A�* 

Average reward per step����|�t6       ��2	���/̩�A�*

epsilon�����U�k.       ��W�	��/̩�A�* 

Average reward per step����K       ��2	ݵ�/̩�A�*

epsilon������{�.       ��W�	II�/̩�A�* 

Average reward per step������a       ��2	<J�/̩�A�*

epsilon������W�.       ��W�	u��/̩�A�* 

Average reward per step����Ћ7O       ��2	)��/̩�A�*

epsilon����K_��.       ��W�	$�$0̩�A�* 

Average reward per step�����V8�       ��2	й$0̩�A�*

epsilon�������H.       ��W�	�O(0̩�A�* 

Average reward per step������       ��2	�P(0̩�A�*

epsilon�����z��.       ��W�	-x*0̩�A�* 

Average reward per step����F�PH       ��2	y*0̩�A�*

epsilon�����l��.       ��W�	m;.0̩�A�* 

Average reward per step����h�ߞ       ��2	�<.0̩�A�*

epsilon�����Fj�.       ��W�	/�/0̩�A�* 

Average reward per step�����,�       ��2	�/0̩�A�*

epsilon�����S�Z.       ��W�	�C20̩�A�* 

Average reward per step������X       ��2	�D20̩�A�*

epsilon�����!8.       ��W�	�40̩�A�* 

Average reward per step����/�T�       ��2	_40̩�A�*

epsilon������3\.       ��W�	60̩�A�* 

Average reward per step�����       ��2	�60̩�A�*

epsilon�������.       ��W�	~80̩�A�* 

Average reward per step����{&^       ��2	�~80̩�A�*

epsilon�����Y�c.       ��W�	9�|0̩�A�* 

Average reward per step����m��|       ��2	J�|0̩�A�*

epsilon�����{-�.       ��W�	�w~0̩�A�* 

Average reward per step�����`��       ��2	�x~0̩�A�*

epsilon����h{��.       ��W�	�Ҁ0̩�A�* 

Average reward per step����{���       ��2	�Ӏ0̩�A�*

epsilon����[c��0       ���_	r��0̩�Ad*#
!
Average reward per episodeUU�����I.       ��W�	H��0̩�Ad*!

total reward per episode  �P��.       ��W�		݄0̩�A�* 

Average reward per stepUU�����       ��2	�݄0̩�A�*

epsilonUU��~��.       ��W�	fL�0̩�A�* 

Average reward per stepUU���5�z       ��2	IM�0̩�A�*

epsilonUU��n/e�.       ��W�	���0̩�A�* 

Average reward per stepUU���i0�       ��2	��0̩�A�*

epsilonUU��Gs:r.       ��W�	�d�0̩�A�* 

Average reward per stepUU��$�3       ��2	ke�0̩�A�*

epsilonUU��7g|8.       ��W�	��0̩�A�* 

Average reward per stepUU��,�w�       ��2	��0̩�A�*

epsilonUU��Wlcz.       ��W�	v�0̩�A�* 

Average reward per stepUU����9       ��2	�v�0̩�A�*

epsilonUU���/��.       ��W�	<�0̩�A�* 

Average reward per stepUU��S֋e       ��2	��0̩�A�*

epsilonUU��$�2.       ��W�	1�0̩�A�* 

Average reward per stepUU��I�Ϝ       ��2	��0̩�A�*

epsilonUU���Z�.       ��W�	O�0̩�A�* 

Average reward per stepUU���v��       ��2	�0̩�A�*

epsilonUU���{�.       ��W�	b��0̩�A�* 

Average reward per stepUU��D�       ��2	��0̩�A�*

epsilonUU��0͞5.       ��W�	�4�0̩�A�* 

Average reward per stepUU��A��Z       ��2	�5�0̩�A�*

epsilonUU��sm�.       ��W�	�0̩�A�* 

Average reward per stepUU����a       ��2	�0̩�A�*

epsilonUU��uv .       ��W�	�0̩�A�* 

Average reward per stepUU��4��       ��2	��0̩�A�*

epsilonUU��T��.       ��W�	ʧ0̩�A�* 

Average reward per stepUU���+��       ��2	�ʧ0̩�A�*

epsilonUU����q�.       ��W�	C��0̩�A�* 

Average reward per stepUU�����A       ��2	���0̩�A�*

epsilonUU��\L8.       ��W�	gӫ0̩�A�* 

Average reward per stepUU��n���       ��2	ԫ0̩�A�*

epsilonUU�����.       ��W�	�<�0̩�A�* 

Average reward per stepUU��
l       ��2	�=�0̩�A�*

epsilonUU����`�.       ��W�	Zh�0̩�A�* 

Average reward per stepUU��4i�       ��2	�i�0̩�A�*

epsilonUU��b�&�.       ��W�	��0̩�A�* 

Average reward per stepUU��Q�       ��2	���0̩�A�*

epsilonUU���F>�.       ��W�	u�0̩�A�* 

Average reward per stepUU���kߑ       ��2	d�0̩�A�*

epsilonUU�����@.       ��W�	�o�0̩�A�* 

Average reward per stepUU��'�u_       ��2	�p�0̩�A�*

epsilonUU����.       ��W�	g׼0̩�A�* 

Average reward per stepUU�����       ��2	bؼ0̩�A�*

epsilonUU���c.       ��W�	V�0̩�A�* 

Average reward per stepUU����"|       ��2	J�0̩�A�*

epsilonUU����G4.       ��W�	��0̩�A�* 

Average reward per stepUU��֎��       ��2		��0̩�A�*

epsilonUU���J��0       ���_	t��0̩�Ae*#
!
Average reward per episode����>e�d.       ��W�	>��0̩�Ae*!

total reward per episode  !�>S��.       ��W�	1�0̩�A�* 

Average reward per step����c��       ��2	��0̩�A�*

epsilon����f^j[.       ��W�	���0̩�A�* 

Average reward per step�����0�       ��2	o��0̩�A�*

epsilon����ɏ\�.       ��W�	��0̩�A�* 

Average reward per step�������<       ��2	��0̩�A�*

epsilon�����M5�.       ��W�	���0̩�A�* 

Average reward per step�����V��       ��2	���0̩�A�*

epsilon����=��2.       ��W�	�B�0̩�A�* 

Average reward per step�����?+       ��2	RC�0̩�A�*

epsilon�����i�&.       ��W�	`��0̩�A�* 

Average reward per step�����M��       ��2	`��0̩�A�*

epsilon����iO.       ��W�	 $�0̩�A�* 

Average reward per step������i       ��2	%�0̩�A�*

epsilon����t���.       ��W�	&��0̩�A�* 

Average reward per step�����}j       ��2	��0̩�A�*

epsilon�����˛.       ��W�	K��0̩�A�* 

Average reward per step����2� �       ��2	y��0̩�A�*

epsilon����,KX.       ��W�	&s�0̩�A�* 

Average reward per step�������)       ��2	t�0̩�A�*

epsilon�����F�.       ��W�	%��0̩�A�* 

Average reward per step���� My}       ��2	���0̩�A�*

epsilon�����Q�V.       ��W�	���0̩�A�* 

Average reward per step����nh,       ��2	���0̩�A�*

epsilon��������.       ��W�	�+�0̩�A�* 

Average reward per step�����W�       ��2	�,�0̩�A�*

epsilon����L���.       ��W�	��0̩�A�* 

Average reward per step����rp��       ��2	*�0̩�A�*

epsilon����啱.       ��W�	p��0̩�A�* 

Average reward per step����s��       ��2	%��0̩�A�*

epsilon����WhV.       ��W�	7�0̩�A�* 

Average reward per step�������       ��2	"�0̩�A�*

epsilon������.       ��W�	���0̩�A�* 

Average reward per step����_���       ��2	a��0̩�A�*

epsilon����k�$�.       ��W�	�!�0̩�A�* 

Average reward per step����Lnd!       ��2	�"�0̩�A�*

epsilon������b.       ��W�	�!�0̩�A�* 

Average reward per step�����t�y       ��2	�"�0̩�A�*

epsilon����C�.�.       ��W�	�0̩�A�* 

Average reward per step�������<       ��2	�0̩�A�*

epsilon�����!��.       ��W�	�Q�0̩�A�* 

Average reward per step�������       ��2	�R�0̩�A�*

epsilon����&$x�.       ��W�	[?�0̩�A�* 

Average reward per step������П       ��2	h@�0̩�A�*

epsilon����Kru�.       ��W�	���0̩�A�* 

Average reward per step��������       ��2	��0̩�A�*

epsilon����}�.       ��W�	�A�0̩�A�* 

Average reward per step����N���       ��2	�B�0̩�A�*

epsilon������e.       ��W�	��1̩�A�* 

Average reward per step����r��       ��2	��1̩�A�*

epsilon��������.       ��W�	�1̩�A�* 

Average reward per step�����۟       ��2	�1̩�A�*

epsilon����y���.       ��W�	��1̩�A�* 

Average reward per step������X�       ��2	��1̩�A�*

epsilon����Y���.       ��W�	��
1̩�A�* 

Average reward per step����H�R�       ��2	��
1̩�A�*

epsilon��������.       ��W�	��1̩�A�* 

Average reward per step�����j�	       ��2	U�1̩�A�*

epsilon����HJ	.       ��W�	A1̩�A�* 

Average reward per step������~�       ��2	�B1̩�A�*

epsilon�����t��.       ��W�	+�1̩�A�* 

Average reward per step����g�}       ��2	+�1̩�A�*

epsilon�����H?.       ��W�	�_1̩�A�* 

Average reward per step�����h�>       ��2	�`1̩�A�*

epsilon�����2��.       ��W�	�1̩�A�* 

Average reward per step����|�?       ��2	�1̩�A�*

epsilon����R�g^.       ��W�	 V1̩�A�* 

Average reward per step����r'�-       ��2	W1̩�A�*

epsilon����)��.       ��W�	��!1̩�A�* 

Average reward per step����eE�       ��2	_�!1̩�A�*

epsilon������!�.       ��W�	B#1̩�A�* 

Average reward per step������U�       ��2	 C#1̩�A�*

epsilon�����af(.       ��W�	��%1̩�A�* 

Average reward per step�����       ��2	u�%1̩�A�*

epsilon����lbpr.       ��W�	Yl'1̩�A�* 

Average reward per step�����	�       ��2	@m'1̩�A�*

epsilon����^MY8.       ��W�	ٵ)1̩�A�* 

Average reward per step����;	��       ��2	��)1̩�A�*

epsilon�����[Ǎ.       ��W�	B",1̩�A�* 

Average reward per step�������       ��2	:#,1̩�A�*

epsilon�����<�A.       ��W�	/�/1̩�A�* 

Average reward per step������n       ��2	�/1̩�A�*

epsilon�����,T.       ��W�	ʣ11̩�A�* 

Average reward per step����+�G�       ��2	פ11̩�A�*

epsilon����� �.       ��W�	��31̩�A�* 

Average reward per step����L&��       ��2	��31̩�A�*

epsilon����|�v.       ��W�	�g61̩�A�* 

Average reward per step�����I/R       ��2	'i61̩�A�*

epsilon����#��X.       ��W�	G�:1̩�A�* 

Average reward per step�����v�       ��2	O�:1̩�A�*

epsilon������`�.       ��W�	��1̩�A�* 

Average reward per step�������f       ��2	��1̩�A�*

epsilon����5��
.       ��W�	V�1̩�A�* 

Average reward per step�������       ��2	�V�1̩�A�*

epsilon����b�=�.       ��W�	�ʄ1̩�A�* 

Average reward per step�������J       ��2	l̈́1̩�A�*

epsilon����PT`r.       ��W�	D��1̩�A�* 

Average reward per step�����DJF       ��2	P��1̩�A�*

epsilon�����5!0       ���_	�#�1̩�Af*#
!
Average reward per episode���.�6�.       ��W�	l$�1̩�Af*!

total reward per episode  ���%��.       ��W�	�H�1̩�A�* 

Average reward per step�����'       ��2	�I�1̩�A�*

epsilon���@�)v.       ��W�	��1̩�A�* 

Average reward per step���	(8?       ��2	���1̩�A�*

epsilon���!	��.       ��W�	;�1̩�A�* 

Average reward per step���&�`i       ��2	�;�1̩�A�*

epsilon���{|�.       ��W�	��1̩�A�* 

Average reward per step�����6       ��2	���1̩�A�*

epsilon���e&Z�.       ��W�	�8�1̩�A�* 

Average reward per step����[V       ��2	z9�1̩�A�*

epsilon��忻;�.       ��W�	�1̩�A�* 

Average reward per step�����       ��2	�Ý1̩�A�*

epsilon���RN31.       ��W�	�Z�1̩�A�* 

Average reward per step��� O�       ��2	O[�1̩�A�*

epsilon����M�N.       ��W�	 �1̩�A�* 

Average reward per step��忬g��       ��2	��1̩�A�*

epsilon���.t��.       ��W�	ς�1̩�A�* 

Average reward per step����E�       ��2	Ã�1̩�A�*

epsilon����h.       ��W�	�6�1̩�A�* 

Average reward per step�������       ��2	�7�1̩�A�*

epsilon���h��.       ��W�		q�1̩�A�* 

Average reward per step��忋�:�       ��2	�q�1̩�A�*

epsilon�����-Y.       ��W�	���1̩�A�* 

Average reward per step���
r7       ��2	��1̩�A�*

epsilon���.ֽ�.       ��W�	�	�1̩�A�* 

Average reward per step���B��e       ��2	5
�1̩�A�*

epsilon����	��.       ��W�	�n�1̩�A�* 

Average reward per step���`uw       ��2	�o�1̩�A�*

epsilon���^���.       ��W�	�&�1̩�A�* 

Average reward per step���QK}�       ��2	�'�1̩�A�*

epsilon���z
.       ��W�	��1̩�A�* 

Average reward per step���a�       ��2	��1̩�A�*

epsilon��忬��@.       ��W�	5��1̩�A�* 

Average reward per step��忳44h       ��2	��1̩�A�*

epsilon��忓�^9.       ��W�	`! 2̩�A�* 

Average reward per step����mA       ��2	O" 2̩�A�*

epsilon��忴�V9.       ��W�	��2̩�A�* 

Average reward per step��念?]�       ��2	��2̩�A�*

epsilon���ദ�.       ��W�	;82̩�A�* 

Average reward per step����
�       ��2	?92̩�A�*

epsilon�����v�.       ��W�	W2̩�A�* 

Average reward per step���	��w       ��2	�W2̩�A�*

epsilon���Sq�u.       ��W�	�gK2̩�A�* 

Average reward per step���T� �       ��2	�hK2̩�A�*

epsilon����	�.       ��W�	h�N2̩�A�* 

Average reward per step��忡u��       ��2	%�N2̩�A�*

epsilon��忮e�.       ��W�	�Q2̩�A�* 

Average reward per step���uD�       ��2	�	Q2̩�A�*

epsilon���]�.       ��W�	ofU2̩�A�* 

Average reward per step����ޘD       ��2	�gU2̩�A�*

epsilon���|��.       ��W�	x�X2̩�A�* 

Average reward per step����d�       ��2	��X2̩�A�*

epsilon���/�Z.       ��W�	�l[2̩�A�* 

Average reward per step����z�       ��2	�m[2̩�A�*

epsilon���lk�.       ��W�	�_2̩�A�* 

Average reward per step��忟�d       ��2	n_2̩�A�*

epsilon��忇��o.       ��W�	`a2̩�A�* 

Average reward per step���ԉn       ��2	�`a2̩�A�*

epsilon���y�x�.       ��W�	k'e2̩�A�* 

Average reward per step���C�2       ��2	=(e2̩�A�*

epsilon��忿�6.       ��W�	��f2̩�A�* 

Average reward per step�����       ��2	��f2̩�A�*

epsilon����2.       ��W�	v5i2̩�A�* 

Average reward per step����^��       ��2	i6i2̩�A�*

epsilon���A��.       ��W�	�\k2̩�A�* 

Average reward per step����ev       ��2	p]k2̩�A�*

epsilon����wD.       ��W�	��m2̩�A�* 

Average reward per step��忀��       ��2	̗m2̩�A�*

epsilon���Щ�c.       ��W�	��o2̩�A�* 

Average reward per step���U~7       ��2	��o2̩�A�*

epsilon��忏��1.       ��W�	Ֆq2̩�A�* 

Average reward per step��応�a       ��2	ėq2̩�A�*

epsilon��忋ܻ�.       ��W�	E�s2̩�A�* 

Average reward per step����P�f       ��2	,�s2̩�A�*

epsilon���"�n.       ��W�	��v2̩�A�* 

Average reward per step���'�ڔ       ��2	��v2̩�A�*

epsilon��忢1�$.       ��W�	�~z2̩�A�* 

Average reward per step���Bl�_       ��2	�z2̩�A�*

epsilon������.       ��W�	�~2̩�A�* 

Average reward per step���-���       ��2	`~2̩�A�*

epsilon���!�>4.       ��W�	(��2̩�A�* 

Average reward per step������       ��2	A��2̩�A�*

epsilon���)�{�.       ��W�	6r�2̩�A�* 

Average reward per step��忌��P       ��2	Ks�2̩�A�*

epsilon��忑>�.       ��W�	cb�2̩�A�* 

Average reward per step����x       ��2	Ac�2̩�A�*

epsilon���ھ�'.       ��W�	�2̩�A�* 

Average reward per step���$&)�       ��2	%�2̩�A�*

epsilon����=Y�.       ��W�	���2̩�A�* 

Average reward per step��忈#�x       ��2	���2̩�A�*

epsilon���H�=.       ��W�	�Ď2̩�A�* 

Average reward per step��忼Mid       ��2	�Ŏ2̩�A�*

epsilon����
�m.       ��W�	�v�2̩�A�* 

Average reward per step���aJ.       ��2	}w�2̩�A�*

epsilon�����.       ��W�	(Ւ2̩�A�* 

Average reward per step���Rq7       ��2	֒2̩�A�*

epsilon��忮�Th.       ��W�	��2̩�A�* 

Average reward per step���a�       ��2	�2̩�A�*

epsilon��忣K�.       ��W�	��2̩�A�* 

Average reward per step�����4d       ��2	_�2̩�A�*

epsilon��忡�.       ��W�	:!�2̩�A�* 

Average reward per step����<�l       ��2	�!�2̩�A�*

epsilon���C���.       ��W�	h[�2̩�A�* 

Average reward per step���6č�       ��2	\�2̩�A�*

epsilon��忏���.       ��W�	��2̩�A�* 

Average reward per step������       ��2	��2̩�A�*

epsilon����G0       ���_	��2̩�Ag*#
!
Average reward per episode"5��
��.       ��W�	��2̩�Ag*!

total reward per episode  ����U.       ��W�	�2̩�A�* 

Average reward per step"5�����<       ��2	��2̩�A�*

epsilon"5��iCG�.       ��W�	�j�2̩�A�* 

Average reward per step"5��i��       ��2	�k�2̩�A�*

epsilon"5��&��t.       ��W�	e��2̩�A�* 

Average reward per step"5���p�       ��2	 �2̩�A�*

epsilon"5��S���.       ��W�	K��2̩�A�* 

Average reward per step"5���ݎW       ��2	*��2̩�A�*

epsilon"5���=��.       ��W�	�a�2̩�A�* 

Average reward per step"5����!�       ��2	�b�2̩�A�*

epsilon"5��kןd.       ��W�	���2̩�A�* 

Average reward per step"5���H	       ��2	��2̩�A�*

epsilon"5���^E.       ��W�	h�2̩�A�* 

Average reward per step"5���M$       ��2	�h�2̩�A�*

epsilon"5�����.       ��W�	(��2̩�A�* 

Average reward per step"5���Nk2       ��2	9��2̩�A�*

epsilon"5��$��.       ��W�	��2̩�A�* 

Average reward per step"5��1�4�       ��2	��2̩�A�*

epsilon"5���.^.       ��W�	��2̩�A�* 

Average reward per step"5��H�V       ��2	���2̩�A�*

epsilon"5��s��Y.       ��W�	�R�2̩�A�* 

Average reward per step"5��пa�       ��2	�S�2̩�A�*

epsilon"5���K��.       ��W�	��2̩�A�* 

Average reward per step"5���LT�       ��2	���2̩�A�*

epsilon"5��a�:.       ��W�	�C�2̩�A�* 

Average reward per step"5���+�U       ��2	tD�2̩�A�*

epsilon"5���Y�n.       ��W�	���2̩�A�* 

Average reward per step"5����a       ��2	���2̩�A�*

epsilon"5�����.       ��W�	i��2̩�A�* 

Average reward per step"5��l�       ��2	z��2̩�A�*

epsilon"5��s7@.       ��W�	m:�2̩�A�* 

Average reward per step"5���?�D       ��2	�;�2̩�A�*

epsilon"5����].       ��W�	_�2̩�A�* 

Average reward per step"5���Kt       ��2	�_�2̩�A�*

epsilon"5����t�.       ��W�	���2̩�A�* 

Average reward per step"5��e�       ��2	[��2̩�A�*

epsilon"5��v���.       ��W�	���2̩�A�* 

Average reward per step"5����v�       ��2	���2̩�A�*

epsilon"5��g�͓0       ���_	�2̩�Ah*#
!
Average reward per episode�k��+.       ��W�	��2̩�Ah*!

total reward per episode  "�7��1.       ��W�	�{�2̩�A�* 

Average reward per step�k�q�b�       ��2	(~�2̩�A�*

epsilon�k��H��.       ��W�	hw�2̩�A�* 

Average reward per step�k� z>�       ��2	!x�2̩�A�*

epsilon�k�?�~<.       ��W�	���2̩�A�* 

Average reward per step�k�M�7        ��2	s��2̩�A�*

epsilon�k���^.       ��W�	�K�2̩�A�* 

Average reward per step�k���:�       ��2	^L�2̩�A�*

epsilon�k����.       ��W�	=��2̩�A�* 

Average reward per step�k�Qa�q       ��2	��2̩�A�*

epsilon�k����.       ��W�	O��2̩�A�* 

Average reward per step�k�|���       ��2	��2̩�A�*

epsilon�k�'�=.       ��W�	�S�2̩�A�* 

Average reward per step�k�&΂G       ��2	U�2̩�A�*

epsilon�k�"�x.       ��W�	��2̩�A�* 

Average reward per step�k�X�        ��2	���2̩�A�*

epsilon�k�q��=.       ��W�	H��2̩�A�* 

Average reward per step�k���*:       ��2	H��2̩�A�*

epsilon�k�@c[.       ��W�	���2̩�A�* 

Average reward per step�k�1��m       ��2	���2̩�A�*

epsilon�k�>4��.       ��W�	�j�2̩�A�* 

Average reward per step�k��^i�       ��2	8k�2̩�A�*

epsilon�k�5�i�.       ��W�	j��2̩�A�* 

Average reward per step�k�I��       ��2	j��2̩�A�*

epsilon�k��oqy.       ��W�	�L�2̩�A�* 

Average reward per step�k��ڏ       ��2	�M�2̩�A�*

epsilon�k�<�Qr.       ��W�	z��2̩�A�* 

Average reward per step�k��j��       ��2	��2̩�A�*

epsilon�k�O��+.       ��W�	}X 3̩�A�* 

Average reward per step�k��R��       ��2	.Y 3̩�A�*

epsilon�k��%.       ��W�	��3̩�A�* 

Average reward per step�k�� ֟       ��2	ʈ3̩�A�*

epsilon�k����>.       ��W�	F�3̩�A�* 

Average reward per step�k���3�       ��2	�3̩�A�*

epsilon�k�E8�.       ��W�	�93̩�A�* 

Average reward per step�k�#y��       ��2	2:3̩�A�*

epsilon�k�Ί!�.       ��W�	�|
3̩�A�* 

Average reward per step�k��ʁ�       ��2	�}
3̩�A�*

epsilon�k��.       ��W�	��3̩�A�* 

Average reward per step�k��g%�       ��2	b�3̩�A�*

epsilon�k�����.       ��W�	�3̩�A�* 

Average reward per step�k�$�L�       ��2	�3̩�A�*

epsilon�k�a�.       ��W�	Y3̩�A�* 

Average reward per step�k�Ģu�       ��2	Q3̩�A�*

epsilon�k�(�Y*.       ��W�	o3̩�A�* 

Average reward per step�k��a6       ��2	|�3̩�A�*

epsilon�k�_��.       ��W�	�,3̩�A�* 

Average reward per step�k��Ѩ       ��2	Q-3̩�A�*

epsilon�k���].       ��W�	U3̩�A�* 

Average reward per step�k�2όp       ��2	�U3̩�A�*

epsilon�k����7.       ��W�	�3̩�A�* 

Average reward per step�k� [gt       ��2	�3̩�A�*

epsilon�k��;��.       ��W�	l�3̩�A�* 

Average reward per step�k�LLL�       ��2	�3̩�A�*

epsilon�k��כ.       ��W�	]7 3̩�A�* 

Average reward per step�k�Ң��       ��2	8 3̩�A�*

epsilon�k��پ�.       ��W�	
�!3̩�A�* 

Average reward per step�k��&>       ��2	�!3̩�A�*

epsilon�k���(R.       ��W�	��$3̩�A�* 

Average reward per step�k�_�y�       ��2	��$3̩�A�*

epsilon�k����.       ��W�	�Y(3̩�A�* 

Average reward per step�k�bk��       ��2	:Z(3̩�A�*

epsilon�k�-�=&.       ��W�	�*3̩�A�* 

Average reward per step�k��g�+       ��2	��*3̩�A�*

epsilon�k�j���.       ��W�	1.3̩�A�* 

Average reward per step�k���s       ��2	�1.3̩�A�*

epsilon�k�01��.       ��W�	>03̩�A�* 

Average reward per step�k����D       ��2	-	03̩�A�*

epsilon�k�Zws.       ��W�	eU23̩�A�* 

Average reward per step�k��C       ��2	V23̩�A�*

epsilon�k��F�.       ��W�	!"43̩�A�* 

Average reward per step�k�5�w�       ��2	#43̩�A�*

epsilon�k�>�rY.       ��W�	�.73̩�A�* 

Average reward per step�k����H       ��2	Z/73̩�A�*

epsilon�k��my�.       ��W�	��:3̩�A�* 

Average reward per step�k��SD       ��2	��:3̩�A�*

epsilon�k��T��.       ��W�	Bw<3̩�A�* 

Average reward per step�k��`P       ��2	x<3̩�A�*

epsilon�k�v,��.       ��W�	�>3̩�A�* 

Average reward per step�k��Gt�       ��2	�>3̩�A�*

epsilon�k��;�%.       ��W�	ʍ@3̩�A�* 

Average reward per step�k���c       ��2	z�@3̩�A�*

epsilon�k���6�.       ��W�	��B3̩�A�* 

Average reward per step�k�����       ��2	$�B3̩�A�*

epsilon�k����l.       ��W�	-�D3̩�A�* 

Average reward per step�k��qvS       ��2	ݕD3̩�A�*

epsilon�k�}K=}.       ��W�	f�F3̩�A�* 

Average reward per step�k�N���       ��2	+�F3̩�A�*

epsilon�k���.       ��W�	 �I3̩�A�* 

Average reward per step�k�f���       ��2	ԶI3̩�A�*

epsilon�k� �¹.       ��W�	!M3̩�A�* 

Average reward per step�k�P       ��2	�!M3̩�A�*

epsilon�k���Qi.       ��W�	o�N3̩�A�* 

Average reward per step�k�>���       ��2	w�N3̩�A�*

epsilon�k�ز��.       ��W�	Q3̩�A�* 

Average reward per step�k���ݕ       ��2	Q3̩�A�*

epsilon�k�
Ќ�.       ��W�	ۇS3̩�A�* 

Average reward per step�k�B���       ��2	�S3̩�A�*

epsilon�k��'0       ���_	2�S3̩�Ai*#
!
Average reward per episode��応�H�.       ��W�	�S3̩�Ai*!

total reward per episode  �½�r�.       ��W�	�W3̩�A�* 

Average reward per step���"�       ��2	,�W3̩�A�*

epsilon����Y.       ��W�	wZ3̩�A�* 

Average reward per step��忝��       ��2	Z3̩�A�*

epsilon��忸���.       ��W�	�[3̩�A�* 

Average reward per step���*��)       ��2	֍[3̩�A�*

epsilon�����m.       ��W�	��]3̩�A�* 

Average reward per step���~rN       ��2	��]3̩�A�*

epsilon���TS��.       ��W�	�:`3̩�A�* 

Average reward per step�����!t       ��2	�;`3̩�A�*

epsilon��心L1�.       ��W�	d3̩�A�* 

Average reward per step�����{       ��2	&d3̩�A�*

epsilon��忯6R.       ��W�	q�e3̩�A�* 

Average reward per step���U�6       ��2	��e3̩�A�*

epsilon������.       ��W�	F�g3̩�A�* 

Average reward per step�����*       ��2	��g3̩�A�*

epsilon���� 9.       ��W�	,Ij3̩�A�* 

Average reward per step��忳&R       ��2	Jj3̩�A�*

epsilon���VY��.       ��W�	�l3̩�A�* 

Average reward per step����8c       ��2	̛l3̩�A�*

epsilon���"���.       ��W�	��p3̩�A�* 

Average reward per step���7c5�       ��2	P�p3̩�A�*

epsilon���t�.       ��W�	�>t3̩�A�* 

Average reward per step��忥1��       ��2	:?t3̩�A�*

epsilon������.       ��W�	��v3̩�A�* 

Average reward per step�����Z�       ��2	+�v3̩�A�*

epsilon����[��.       ��W�	Ay3̩�A�* 

Average reward per step���E�uQ       ��2	$y3̩�A�*

epsilon���jt��.       ��W�	��|3̩�A�* 

Average reward per step����6}G       ��2	H�|3̩�A�*

epsilon���6��.       ��W�	c�~3̩�A�* 

Average reward per step����_�B       ��2	^�~3̩�A�*

epsilon���7WL.       ��W�	�}�3̩�A�* 

Average reward per step���LTKL       ��2	�~�3̩�A�*

epsilon����^U0       ���_	��3̩�Aj*#
!
Average reward per episode����.       ��W�	l�3̩�Aj*!

total reward per episode  �E��.       ��W�	>�3̩�A�* 

Average reward per step���'6       ��2	d�3̩�A�*

epsilon�o�H�.       ��W�	�3̩�A�* 

Average reward per step��j)n       ��2	��3̩�A�*

epsilon��!�.       ��W�	��3̩�A�* 

Average reward per step�t6B�       ��2	��3̩�A�*

epsilon��6n�.       ��W�	�3̩�A�* 

Average reward per step�@g��       ��2	C�3̩�A�*

epsilon�O@�`.       ��W�	祝3̩�A�* 

Average reward per step����       ��2	G��3̩�A�*

epsilon���|�.       ��W�	x�3̩�A�* 

Average reward per step�=��       ��2	g�3̩�A�*

epsilon�I��.       ��W�	'/�3̩�A�* 

Average reward per step��c0       ��2	Q0�3̩�A�*

epsilon���.       ��W�	峘3̩�A�* 

Average reward per step���N�       ��2	�3̩�A�*

epsilon��E�.       ��W�	���3̩�A�* 

Average reward per step�Q��       ��2	���3̩�A�*

epsilon���y�.       ��W�		N�3̩�A�* 

Average reward per step���J�       ��2	GW�3̩�A�*

epsilon��s2 .       ��W�	�؟3̩�A�* 

Average reward per step�CE�       ��2	Qٟ3̩�A�*

epsilon��P�M.       ��W�	,�3̩�A�* 

Average reward per step�y^p       ��2	��3̩�A�*

epsilon��*O.       ��W�	�;�3̩�A�* 

Average reward per step����H       ��2	y<�3̩�A�*

epsilon��ʗz.       ��W�	X��3̩�A�* 

Average reward per step�l�Yh       ��2	B��3̩�A�*

epsilon��@.       ��W�	�E�3̩�A�* 

Average reward per step�Dd-�       ��2	�F�3̩�A�*

epsilon���`�.       ��W�	k��3̩�A�* 

Average reward per step���       ��2	��3̩�A�*

epsilon�4��.       ��W�	�ڮ3̩�A�* 

Average reward per step�⃹b       ��2	�ۮ3̩�A�*

epsilon�y�A.       ��W�	1}�3̩�A�* 

Average reward per step�]1�       ��2	�~�3̩�A�*

epsilon�-���.       ��W�	/�3̩�A�* 

Average reward per step��ư<       ��2	/0�3̩�A�*

epsilon�2�M.       ��W�	�Z�3̩�A�* 

Average reward per step�Qi�       ��2	�[�3̩�A�*

epsilon��M�.       ��W�	&�3̩�A�* 

Average reward per step��v�y       ��2	��3̩�A�*

epsilon�rY��.       ��W�	��3̩�A�* 

Average reward per step��K��       ��2	>�3̩�A�*

epsilon�v�x.       ��W�	I/�3̩�A�* 

Average reward per step�a�<�       ��2	+0�3̩�A�*

epsilon�b�.       ��W�	<��3̩�A�* 

Average reward per step�a�4       ��2	#��3̩�A�*

epsilon�2��M.       ��W�	�L�3̩�A�* 

Average reward per step���       ��2	�M�3̩�A�*

epsilon�4ɪ.       ��W�	e5�3̩�A�* 

Average reward per step���0�       ��2	v6�3̩�A�*

epsilon��Y4=.       ��W�	�Z�3̩�A�* 

Average reward per step���       ��2	�[�3̩�A�*

epsilon�u�P.       ��W�	���3̩�A�* 

Average reward per step���       ��2	m��3̩�A�*

epsilon��a.       ��W�	nQ�3̩�A�* 

Average reward per step��5�8       ��2	S�3̩�A�*

epsilon��Ʉ.       ��W�	+��3̩�A�* 

Average reward per step���n       ��2	��3̩�A�*

epsilon��v��.       ��W�	��3̩�A�* 

Average reward per step�
܅�       ��2	��3̩�A�*

epsilon�A�!.       ��W�	R�3̩�A�* 

Average reward per step�P�@       ��2	S�3̩�A�*

epsilon���T�.       ��W�	x
�3̩�A�* 

Average reward per step���@       ��2	��3̩�A�*

epsilon���.       ��W�	O��3̩�A�* 

Average reward per step��wu�       ��2	%��3̩�A�*

epsilon���Et.       ��W�	
��3̩�A�* 

Average reward per step�[V�M       ��2	���3̩�A�*

epsilon���[�.       ��W�	�#�3̩�A�* 

Average reward per step�f
�       ��2	�$�3̩�A�*

epsilon���.       ��W�	*��3̩�A�* 

Average reward per step�)KWv       ��2	X��3̩�A�*

epsilon����0       ���_	��3̩�Ak*#
!
Average reward per episodeL�O�N).       ��W�	���3̩�Ak*!

total reward per episode  ����.       ��W�		�3̩�A�* 

Average reward per stepL�O�����       ��2	�3̩�A�*

epsilonL�O��@��.       ��W�	���3̩�A�* 

Average reward per stepL�O�eh��       ��2	l��3̩�A�*

epsilonL�O��h�q.       ��W�	x
�3̩�A�* 

Average reward per stepL�O���k       ��2	 �3̩�A�*

epsilonL�O�KCE'.       ��W�	�B�3̩�A�* 

Average reward per stepL�O��S       ��2	�C�3̩�A�*

epsilonL�O����.       ��W�	+��3̩�A�* 

Average reward per stepL�O�s�/       ��2	4��3̩�A�*

epsilonL�O�b
.       ��W�	pC�3̩�A�* 

Average reward per stepL�O���ܩ       ��2	 D�3̩�A�*

epsilonL�O��?�4.       ��W�	�p�3̩�A�* 

Average reward per stepL�O��Ԉ�       ��2	�q�3̩�A�*

epsilonL�O�H�K�.       ��W�	���3̩�A�* 

Average reward per stepL�O��7�       ��2	���3̩�A�*

epsilonL�O�Z�3�.       ��W�	��3̩�A�* 

Average reward per stepL�O�m�#�       ��2	��3̩�A�*

epsilonL�O�I��f.       ��W�	�V4̩�A�* 

Average reward per stepL�O�5kh�       ��2	�W4̩�A�*

epsilonL�O���}�.       ��W�		�4̩�A�* 

Average reward per stepL�O�ާ�       ��2	�4̩�A�*

epsilonL�O�!��.       ��W�	�w	4̩�A�* 

Average reward per stepL�O�ST,�       ��2	lx	4̩�A�*

epsilonL�O�J&q.       ��W�	�4̩�A�* 

Average reward per stepL�O�sJQ:       ��2	>	4̩�A�*

epsilonL�O�3��Q.       ��W�	d�4̩�A�* 

Average reward per stepL�O�}b	�       ��2	t�4̩�A�*

epsilonL�O����.       ��W�	��4̩�A�* 

Average reward per stepL�O�R&��       ��2	�4̩�A�*

epsilonL�O�!��%.       ��W�	I�4̩�A�* 

Average reward per stepL�O��Z�6       ��2	�4̩�A�*

epsilonL�O��:�_.       ��W�	&s4̩�A�* 

Average reward per stepL�O�z��       ��2	t4̩�A�*

epsilonL�O�<P�H.       ��W�	1�4̩�A�* 

Average reward per stepL�O��8,m       ��2	g�4̩�A�*

epsilonL�O��#ڐ.       ��W�	��4̩�A�* 

Average reward per stepL�O��}#�       ��2	J�4̩�A�*

epsilonL�O��9.       ��W�	�y4̩�A�* 

Average reward per stepL�O�5yY�       ��2	�z4̩�A�*

epsilonL�O���q.       ��W�	��!4̩�A�* 

Average reward per stepL�O��XFb       ��2	k�!4̩�A�*

epsilonL�O�k��.       ��W�	!#$4̩�A�* 

Average reward per stepL�O���9       ��2	�#$4̩�A�*

epsilonL�O��O�~.       ��W�	#�%4̩�A�* 

Average reward per stepL�O��,��       ��2	��%4̩�A�*

epsilonL�O��2�s.       ��W�	c(4̩�A�* 

Average reward per stepL�O�{g�i       ��2	S	(4̩�A�*

epsilonL�O��
_!.       ��W�	�g*4̩�A�* 

Average reward per stepL�O�଄�       ��2	i*4̩�A�*

epsilonL�O����.       ��W�	;�+4̩�A�* 

Average reward per stepL�O�����       ��2	�+4̩�A�*

epsilonL�O�:U��.       ��W�	�X.4̩�A�* 

Average reward per stepL�O��O�       ��2	�Y.4̩�A�*

epsilonL�O�i���.       ��W�	��/4̩�A�* 

Average reward per stepL�O��* �       ��2	��/4̩�A�*

epsilonL�O��e�.       ��W�	[_24̩�A�* 

Average reward per stepL�O�.�0       ��2	g`24̩�A�*

epsilonL�O��\2.       ��W�	9�44̩�A�* 

Average reward per stepL�O���Խ       ��2	�44̩�A�*

epsilonL�O�Z�4>.       ��W�	)64̩�A�* 

Average reward per stepL�O�<���       ��2	�)64̩�A�*

epsilonL�O�ܢ].       ��W�	ǝ84̩�A�* 

Average reward per stepL�O���Ls       ��2	��84̩�A�*

epsilonL�O��o�0       ���_	L�84̩�Al*#
!
Average reward per episode  ����A�.       ��W�	3�84̩�Al*!

total reward per episode  �`�Ï.       ��W�	6�>4̩�A�* 

Average reward per step  ���_?       ��2	B�>4̩�A�*

epsilon  ����L.       ��W�	en@4̩�A�* 

Average reward per step  ��ܐ*       ��2	o@4̩�A�*

epsilon  ���*�.       ��W�	�B4̩�A�* 

Average reward per step  ���Ĵ�       ��2	ǡB4̩�A�*

epsilon  ����e�.       ��W�	��D4̩�A�* 

Average reward per step  ����       ��2	��D4̩�A�*

epsilon  ���ټ�.       ��W�	̚F4̩�A�* 

Average reward per step  ���/��       ��2	��F4̩�A�*

epsilon  ����?.       ��W�	�4I4̩�A�* 

Average reward per step  �����       ��2	�5I4̩�A�*

epsilon  ����v.       ��W�	��L4̩�A�* 

Average reward per step  ����'�       ��2	��L4̩�A�*

epsilon  ��⮲.       ��W�	�O4̩�A�* 

Average reward per step  ��
��       ��2	qO4̩�A�*

epsilon  ����R�.       ��W�	�9Q4̩�A�* 

Average reward per step  ���F��       ��2	y:Q4̩�A�*

epsilon  ����y.       ��W�	��R4̩�A�* 

Average reward per step  �����	       ��2	��R4̩�A�*

epsilon  ��0�n�.       ��W�	�2U4̩�A�* 

Average reward per step  ��=��       ��2	{3U4̩�A�*

epsilon  ������.       ��W�	=�W4̩�A�* 

Average reward per step  ��Y�K       ��2	Z�W4̩�A�*

epsilon  ��!3B=.       ��W�	��Y4̩�A�* 

Average reward per step  ���=s�       ��2	.�Y4̩�A�*

epsilon  ���Q_.       ��W�	Xq]4̩�A�* 

Average reward per step  ����y�       ��2	Gr]4̩�A�*

epsilon  ��O��.       ��W�	�1_4̩�A�* 

Average reward per step  ���͒�       ��2	�2_4̩�A�*

epsilon  ��GG\�.       ��W�	}�a4̩�A�* 

Average reward per step  ����	z       ��2	ėa4̩�A�*

epsilon  ���P2I.       ��W�	�Kc4̩�A�* 

Average reward per step  ����^�       ��2	wLc4̩�A�*

epsilon  ���y:�.       ��W�	f4̩�A�* 

Average reward per step  ����       ��2	�f4̩�A�*

epsilon  ��*(��.       ��W�	ʥi4̩�A�* 

Average reward per step  ���'�       ��2	��i4̩�A�*

epsilon  ��n�j.       ��W�	yk4̩�A�* 

Average reward per step  ��� �       ��2	k4̩�A�*

epsilon  �����.       ��W�	��m4̩�A�* 

Average reward per step  ��s��'       ��2	��m4̩�A�*

epsilon  ��)��p.       ��W�	j�q4̩�A�* 

Average reward per step  �����i       ��2	�q4̩�A�*

epsilon  ����..       ��W�	��s4̩�A�* 

Average reward per step  ���wzh       ��2	��s4̩�A�*

epsilon  ��h�~�0       ���_	Pt4̩�Am*#
!
Average reward per episode�B����.       ��W�	�t4̩�Am*!

total reward per episode  úa��.       ��W�	��w4̩�A�* 

Average reward per step�B������       ��2	��w4̩�A�*

epsilon�B��G�.       ��W�	�y4̩�A�* 

Average reward per step�B�����       ��2	ظy4̩�A�*

epsilon�B��X�.       ��W�	i|4̩�A�* 

Average reward per step�B��r9�       ��2	\|4̩�A�*

epsilon�B���0޾.       ��W�	��}4̩�A�* 

Average reward per step�B��dI�q       ��2	��}4̩�A�*

epsilon�B��Q�@.       ��W�	�)�4̩�A�* 

Average reward per step�B��i��p       ��2	�*�4̩�A�*

epsilon�B���I3z.       ��W�	.�4̩�A�* 

Average reward per step�B���5r~       ��2	��4̩�A�*

epsilon�B���a�.       ��W�	S��4̩�A�* 

Average reward per step�B���(�#       ��2	��4̩�A�*

epsilon�B��4F=�.       ��W�	"ƈ4̩�A�* 

Average reward per step�B��BYC       ��2	&ǈ4̩�A�*

epsilon�B������.       ��W�	�t�4̩�A�* 

Average reward per step�B��k5U       ��2	�u�4̩�A�*

epsilon�B��8�o.       ��W�	#ݎ4̩�A�* 

Average reward per step�B����8�       ��2	/ގ4̩�A�*

epsilon�B��ă{�.       ��W�	��4̩�A�* 

Average reward per step�B�� �       ��2	ˠ�4̩�A�*

epsilon�B��h��.       ��W�	�>�4̩�A�* 

Average reward per step�B��X��       ��2	�?�4̩�A�*

epsilon�B��?��.       ��W�	W
�4̩�A�* 

Average reward per step�B���>�2       ��2	�
�4̩�A�*

epsilon�B���b%�.       ��W�	ˆ�4̩�A�* 

Average reward per step�B�����	       ��2	f��4̩�A�*

epsilon�B��J���.       ��W�	s�4̩�A�* 

Average reward per step�B��ݥN�       ��2	��4̩�A�*

epsilon�B���f��.       ��W�	Vc�4̩�A�* 

Average reward per step�B��`gz       ��2	kd�4̩�A�*

epsilon�B��T���.       ��W�	C�4̩�A�* 

Average reward per step�B���؆M       ��2	%�4̩�A�*

epsilon�B��~��0       ���_	28�4̩�An*#
!
Average reward per episode��dK�.       ��W�	"9�4̩�An*!

total reward per episode  (à���.       ��W�	��4̩�A�* 

Average reward per step��%�       ��2	��4̩�A�*

epsilon���n.       ��W�	x��4̩�A�* 

Average reward per step�̟�4       ��2	���4̩�A�*

epsilon��@��.       ��W�	�x�4̩�A�* 

Average reward per step�;#a�       ��2	%y�4̩�A�*

epsilon��I�.       ��W�	q̱4̩�A�* 

Average reward per step�4�
4       ��2	Wͱ4̩�A�*

epsilon�x�	�.       ��W�	V��4̩�A�* 

Average reward per step���Cn       ��2	j��4̩�A�*

epsilon�'��\.       ��W�	]��4̩�A�* 

Average reward per step��^��       ��2	T��4̩�A�*

epsilon�+�37.       ��W�	P�4̩�A�* 

Average reward per step�^ayt       ��2		�4̩�A�*

epsilon����K.       ��W�	ѻ4̩�A�* 

Average reward per step��э�       ��2	=һ4̩�A�*

epsilon�����.       ��W�	"3�4̩�A�* 

Average reward per step�8��5       ��2	4�4̩�A�*

epsilon�l��.       ��W�	nh�4̩�A�* 

Average reward per step���!       ��2	�i�4̩�A�*

epsilon��D$.       ��W�	� �4̩�A�* 

Average reward per step��#��       ��2	�!�4̩�A�*

epsilon��h�].       ��W�	+n�4̩�A�* 

Average reward per step�����       ��2	ao�4̩�A�*

epsilon�2(�.       ��W�	N��4̩�A�* 

Average reward per step�����       ��2	,��4̩�A�*

epsilon��5�.       ��W�	��4̩�A�* 

Average reward per step��5       ��2	'��4̩�A�*

epsilon�)�0       ���_	��4̩�Ao*#
!
Average reward per episode%I:�O.a+.       ��W�	`�4̩�Ao*!

total reward per episode  #ü l.       ��W�	��5̩�A�* 

Average reward per step%I:��ut�       ��2	G�5̩�A�*

epsilon%I:�+��.       ��W�	�Q5̩�A�* 

Average reward per step%I:���l       ��2	�R5̩�A�*

epsilon%I:���.       ��W�	��5̩�A�* 

Average reward per step%I:�y       ��2	��5̩�A�*

epsilon%I:�*��9.       ��W�	#�5̩�A�* 

Average reward per step%I:���6�       ��2	/�5̩�A�*

epsilon%I:�v4�.       ��W�	��5̩�A�* 

Average reward per step%I:���#�       ��2	C�5̩�A�*

epsilon%I:���.       ��W�	��5̩�A�* 

Average reward per step%I:�]A/:       ��2	��5̩�A�*

epsilon%I:����.       ��W�	��!5̩�A�* 

Average reward per step%I:�����       ��2	 "5̩�A�*

epsilon%I:��ڴt.       ��W�	�i%5̩�A�* 

Average reward per step%I:�C�4�       ��2	fj%5̩�A�*

epsilon%I:�3]�..       ��W�	��'5̩�A�* 

Average reward per step%I:���X�       ��2	��'5̩�A�*

epsilon%I:��5.       ��W�	hY)5̩�A�* 

Average reward per step%I:��[       ��2	�Z)5̩�A�*

epsilon%I:��.�>.       ��W�	t�+5̩�A�* 

Average reward per step%I:��3�{       ��2	F�+5̩�A�*

epsilon%I:�$�g.       ��W�	�/5̩�A�* 

Average reward per step%I:�EMϩ       ��2	��/5̩�A�*

epsilon%I:��M�.       ��W�	�15̩�A�* 

Average reward per step%I:���-!       ��2	��15̩�A�*

epsilon%I:���6�.       ��W�	��35̩�A�* 

Average reward per step%I:�ڴ       ��2	[�35̩�A�*

epsilon%I:�y}��.       ��W�	o�55̩�A�* 

Average reward per step%I:��S,R       ��2	�55̩�A�*

epsilon%I:����0       ���_	�55̩�Ap*#
!
Average reward per episode1����.       ��W�	h�55̩�Ap*!

total reward per episode  &��i�.       ��W�	�:5̩�A�* 

Average reward per step1��b;�       ��2	Z:5̩�A�*

epsilon1��8�1.       ��W�	�+<5̩�A�* 

Average reward per step1��V/�       ��2	s,<5̩�A�*

epsilon1��O�.       ��W�	�>5̩�A�* 

Average reward per step1�Tu��       ��2	�>5̩�A�*

epsilon1����.       ��W�	��?5̩�A�* 

Average reward per step1�_�%       ��2	��?5̩�A�*

epsilon1�(5a`.       ��W�	�6B5̩�A�* 

Average reward per step1����       ��2	�7B5̩�A�*

epsilon1���Z.       ��W�	��C5̩�A�* 

Average reward per step1�uG;_       ��2	u�C5̩�A�*

epsilon1�ڦ�B.       ��W�	:F5̩�A�* 

Average reward per step1��/[�       ��2	�:F5̩�A�*

epsilon1��W`|.       ��W�	��H5̩�A�* 

Average reward per step1��)}�       ��2	E�H5̩�A�*

epsilon1�x'.       ��W�	�gL5̩�A�* 

Average reward per step1�8�X       ��2	�hL5̩�A�*

epsilon1��bѶ.       ��W�	�M5̩�A�* 

Average reward per step1���q�       ��2	��M5̩�A�*

epsilon1���8.       ��W�	RP5̩�A�* 

Average reward per step1���>)       ��2		SP5̩�A�*

epsilon1�ҌJ.       ��W�	v4S5̩�A�* 

Average reward per step1� �E�       ��2	"5S5̩�A�*

epsilon1�s��.       ��W�	2�V5̩�A�* 

Average reward per step1�1C1       ��2	�V5̩�A�*

epsilon1���>6.       ��W�	�5Y5̩�A�* 

Average reward per step1�rQ}       ��2	z6Y5̩�A�*

epsilon1��#2.       ��W�	�7]5̩�A�* 

Average reward per step1��pTy       ��2	~8]5̩�A�*

epsilon1�2�.       ��W�	��^5̩�A�* 

Average reward per step1�a�6       ��2	f�^5̩�A�*

epsilon1��v��.       ��W�	��`5̩�A�* 

Average reward per step1��d       ��2	��`5̩�A�*

epsilon1��a�n.       ��W�	�c5̩�A�* 

Average reward per step1��He�       ��2	Oc5̩�A�*

epsilon1� ��.       ��W�	/1e5̩�A�* 

Average reward per step1����       ��2	2e5̩�A�*

epsilon1����.       ��W�	v�g5̩�A�* 

Average reward per step1�|��       ��2	"�g5̩�A�*

epsilon1�ف��.       ��W�	��i5̩�A�* 

Average reward per step1�P�#       ��2	��i5̩�A�*

epsilon1�~�.       ��W�	�vm5̩�A�* 

Average reward per step1���EE       ��2	�wm5̩�A�*

epsilon1��u��.       ��W�	�o5̩�A�* 

Average reward per step1���ft       ��2	ʨo5̩�A�*

epsilon1���P�.       ��W�	@/q5̩�A�* 

Average reward per step1���	,       ��2	�/q5̩�A�*

epsilon1�P��".       ��W�	�ys5̩�A�* 

Average reward per step1�R���       ��2	pzs5̩�A�*

epsilon1�և(�.       ��W�	ٯu5̩�A�* 

Average reward per step1��V/�       ��2	}�u5̩�A�*

epsilon1�2D�.       ��W�	"Pw5̩�A�* 

Average reward per step1�7K	       ��2	]Qw5̩�A�*

epsilon1��+o].       ��W�	K�y5̩�A�* 

Average reward per step1�c=�       ��2	��y5̩�A�*

epsilon1�#��.       ��W�	X�{5̩�A�* 

Average reward per step1�58�       ��2	i�{5̩�A�*

epsilon1����.       ��W�	�~5̩�A�* 

Average reward per step1����0       ��2	�~5̩�A�*

epsilon1�Z+Nx.       ��W�	���5̩�A�* 

Average reward per step1�P`�       ��2	���5̩�A�*

epsilon1�E�*�.       ��W�	Ւ�5̩�A�* 

Average reward per step1����       ��2	ٓ�5̩�A�*

epsilon1�_x�.       ��W�	�(�5̩�A�* 

Average reward per step1���_       ��2	R)�5̩�A�*

epsilon1��-�+.       ��W�	<��5̩�A�* 

Average reward per step1���r       ��2	��5̩�A�*

epsilon1��q.       ��W�	��5̩�A�* 

Average reward per step1��V��       ��2	��5̩�A�*

epsilon1��95�.       ��W�	�7�5̩�A�* 

Average reward per step1�j��r       ��2	~8�5̩�A�*

epsilon1�S��.       ��W�	�x�5̩�A�* 

Average reward per step1�\K^       ��2	�y�5̩�A�*

epsilon1�ǅE.       ��W�	���5̩�A�* 

Average reward per step1�?�~�       ��2	̵�5̩�A�*

epsilon1��j�.       ��W�	���5̩�A�* 

Average reward per step1�uR��       ��2	5��5̩�A�*

epsilon1�B��.       ��W�	K�5̩�A�* 

Average reward per step1�7���       ��2	��5̩�A�*

epsilon1�w��.       ��W�	9D�5̩�A�* 

Average reward per step1�=f�       ��2	�D�5̩�A�*

epsilon1�Z��e.       ��W�	�#�5̩�A�* 

Average reward per step1�M�M       ��2	�$�5̩�A�*

epsilon1��G=.       ��W�	ۊ�5̩�A�* 

Average reward per step1�ݏ
       ��2	���5̩�A�*

epsilon1�bh��.       ��W�	6�5̩�A�* 

Average reward per step1��Ԡ       ��2	��5̩�A�*

epsilon1��mL.       ��W�	��5̩�A�* 

Average reward per step1��R�       ��2	���5̩�A�*

epsilon1��߃.       ��W�	k��5̩�A�* 

Average reward per step1�:�Nt       ��2	s��5̩�A�*

epsilon1� q�W.       ��W�	�r�5̩�A�* 

Average reward per step1�@oY       ��2	�v�5̩�A�*

epsilon1�[G�.       ��W�	 �5̩�A�* 

Average reward per step1�β�Q       ��2	!�5̩�A�*

epsilon1�����.       ��W�	z��5̩�A�* 

Average reward per step1��kR�       ��2	e��5̩�A�*

epsilon1����	.       ��W�	��5̩�A�* 

Average reward per step1����}       ��2	��5̩�A�*

epsilon1���6�.       ��W�	5}�5̩�A�* 

Average reward per step1��;��       ��2	�}�5̩�A�*

epsilon1�M�*.       ��W�	G��5̩�A�* 

Average reward per step1��
��       ��2	B��5̩�A�*

epsilon1�f!l�.       ��W�	�r�5̩�A�* 

Average reward per step1�d*�       ��2	 t�5̩�A�*

epsilon1�1���.       ��W�	Ƚ5̩�A�* 

Average reward per step1��"��       ��2	ɽ5̩�A�*

epsilon1����F.       ��W�	}�5̩�A�* 

Average reward per step1�r#;L       ��2	d�5̩�A�*

epsilon1�|��.       ��W�	���5̩�A�* 

Average reward per step1��k�       ��2	��5̩�A�*

epsilon1����0       ���_	���5̩�Aq*#
!
Average reward per episode�m;����.       ��W�	@��5̩�Aq*!

total reward per episode  $��, 8.       ��W�	�g�5̩�A�* 

Average reward per step�m;��e�       ��2	{h�5̩�A�*

epsilon�m;��qL.       ��W�	K#�5̩�A�* 

Average reward per step�m;�9,��       ��2	[$�5̩�A�*

epsilon�m;��D$y.       ��W�	���5̩�A�* 

Average reward per step�m;��Y$>       ��2	���5̩�A�*

epsilon�m;��Oj�.       ��W�	^L�5̩�A�* 

Average reward per step�m;���--       ��2	<M�5̩�A�*

epsilon�m;����S.       ��W�	��5̩�A�* 

Average reward per step�m;�q�eV       ��2	>�5̩�A�*

epsilon�m;��%.       ��W�	 ^�5̩�A�* 

Average reward per step�m;���
0       ��2	$_�5̩�A�*

epsilon�m;�s ��.       ��W�	N'�5̩�A�* 

Average reward per step�m;���       ��2	((�5̩�A�*

epsilon�m;��&s�.       ��W�	�T�5̩�A�* 

Average reward per step�m;�6��       ��2	�U�5̩�A�*

epsilon�m;��1ٺ.       ��W�	���5̩�A�* 

Average reward per step�m;���_       ��2	���5̩�A�*

epsilon�m;���.       ��W�	r�5̩�A�* 

Average reward per step�m;��h6       ��2	2s�5̩�A�*

epsilon�m;��Es.       ��W�	��5̩�A�* 

Average reward per step�m;�6�*�       ��2	ҩ�5̩�A�*

epsilon�m;�bZư.       ��W�	f��5̩�A�* 

Average reward per step�m;�N�       ��2	n��5̩�A�*

epsilon�m;���ڳ.       ��W�	\�5̩�A�* 

Average reward per step�m;���	�       ��2	��5̩�A�*

epsilon�m;�� �.       ��W�	?��5̩�A�* 

Average reward per step�m;��Yw       ��2	���5̩�A�*

epsilon�m;�g�.�.       ��W�	a4�5̩�A�* 

Average reward per step�m;�.3<E       ��2	i5�5̩�A�*

epsilon�m;�����.       ��W�	��5̩�A�* 

Average reward per step�m;�y۶�       ��2	��5̩�A�*

epsilon�m;��Z��.       ��W�	}�5̩�A�* 

Average reward per step�m;���n�       ��2	�}�5̩�A�*

epsilon�m;����v.       ��W�	�:�5̩�A�* 

Average reward per step�m;�e�^       ��2	~;�5̩�A�*

epsilon�m;����.       ��W�	��5̩�A�* 

Average reward per step�m;��]       ��2	��5̩�A�*

epsilon�m;����T.       ��W�	^��5̩�A�* 

Average reward per step�m;����       ��2	��5̩�A�*

epsilon�m;�?eL%.       ��W�	4��5̩�A�* 

Average reward per step�m;���       ��2	M��5̩�A�*

epsilon�m;�fSb&.       ��W�	���5̩�A�* 

Average reward per step�m;����;       ��2	���5̩�A�*

epsilon�m;��&�.       ��W�	�t6̩�A�* 

Average reward per step�m;��U,       ��2	uu6̩�A�*

epsilon�m;����s.       ��W�	�6̩�A�* 

Average reward per step�m;��զ       ��2	�6̩�A�*

epsilon�m;�<�|�.       ��W�	�^6̩�A�* 

Average reward per step�m;�z9k�       ��2	�_6̩�A�*

epsilon�m;����.       ��W�	�6̩�A�* 

Average reward per step�m;�ܪ�J       ��2	�6̩�A�*

epsilon�m;��ot3.       ��W�	��6̩�A�* 

Average reward per step�m;�3��       ��2	c�6̩�A�*

epsilon�m;���{�.       ��W�	$b6̩�A�* 

Average reward per step�m;��E,       ��2	�b6̩�A�*

epsilon�m;��.       ��W�		�6̩�A�* 

Average reward per step�m;��h��       ��2	�6̩�A�*

epsilon�m;��$KC.       ��W�	�n6̩�A�* 

Average reward per step�m;��9�       ��2	�o6̩�A�*

epsilon�m;�'*�.       ��W�	��6̩�A�* 

Average reward per step�m;�-���       ��2	3�6̩�A�*

epsilon�m;����1.       ��W�	_\6̩�A�* 

Average reward per step�m;���æ       ��2	)]6̩�A�*

epsilon�m;�O�y�.       ��W�	U�6̩�A�* 

Average reward per step�m;��y!       ��2	L�6̩�A�*

epsilon�m;��.L�.       ��W�	S$!6̩�A�* 

Average reward per step�m;����       ��2	J'!6̩�A�*

epsilon�m;�#���.       ��W�	Զ$6̩�A�* 

Average reward per step�m;��^Z       ��2	��$6̩�A�*

epsilon�m;�����.       ��W�	��&6̩�A�* 

Average reward per step�m;�a�       ��2	�&6̩�A�*

epsilon�m;�t��.       ��W�	��(6̩�A�* 

Average reward per step�m;���@       ��2	�(6̩�A�*

epsilon�m;�49:�.       ��W�	?�*6̩�A�* 

Average reward per step�m;��1L�       ��2	�*6̩�A�*

epsilon�m;���.       ��W�	�.6̩�A�* 

Average reward per step�m;��du�       ��2	�.6̩�A�*

epsilon�m;�#�d�.       ��W�	3�16̩�A�* 

Average reward per step�m;�j8�k       ��2	��16̩�A�*

epsilon�m;�u>�.       ��W�	��56̩�A�* 

Average reward per step�m;��U�       ��2	}�56̩�A�*

epsilon�m;����.       ��W�	K�96̩�A�* 

Average reward per step�m;�RY"       ��2	C�96̩�A�*

epsilon�m;��Vhk0       ���_	4�96̩�Ar*#
!
Average reward per episode�$)�t�|�.       ��W�	ܹ96̩�Ar*!

total reward per episode  ��g��.       ��W�	w1>6̩�A�* 

Average reward per step�$)�� �       ��2	j2>6̩�A�*

epsilon�$)�A�xE.       ��W�	vB6̩�A�* 

Average reward per step�$)��;Բ       ��2	�vB6̩�A�*

epsilon�$)�x��.       ��W�	&�E6̩�A�* 

Average reward per step�$)�%�Ov       ��2	D�E6̩�A�*

epsilon�$)�^���.       ��W�	��H6̩�A�* 

Average reward per step�$)�K�R       ��2	��H6̩�A�*

epsilon�$)��\L.       ��W�	�L6̩�A�* 

Average reward per step�$)�Wr�$       ��2	L6̩�A�*

epsilon�$)���.       ��W�	'/N6̩�A�* 

Average reward per step�$)�r��J       ��2	<0N6̩�A�*

epsilon�$)�pr�w.       ��W�	_P6̩�A�* 

Average reward per step�$)��;Q�       ��2	�P6̩�A�*

epsilon�$)�r�{.       ��W�	zQR6̩�A�* 

Average reward per step�$)����       ��2	�RR6̩�A�*

epsilon�$)�`�'�.       ��W�	�PT6̩�A�* 

Average reward per step�$)�Y}F�       ��2	�QT6̩�A�*

epsilon�$)���Њ.       ��W�	^�V6̩�A�* 

Average reward per step�$)��8�       ��2	�V6̩�A�*

epsilon�$)��gj�.       ��W�	mVZ6̩�A�* 

Average reward per step�$)�Ld�       ��2	6WZ6̩�A�*

epsilon�$)�D�].       ��W�	��\6̩�A�* 

Average reward per step�$)�N���       ��2	��\6̩�A�*

epsilon�$)�HS`�.       ��W�	�a^6̩�A�* 

Average reward per step�$)�7E�       ��2	�b^6̩�A�*

epsilon�$)��*�.       ��W�	ʣ`6̩�A�* 

Average reward per step�$)��{O�       ��2	@�`6̩�A�*

epsilon�$)�͒C.       ��W�	�c6̩�A�* 

Average reward per step�$)�Y�       ��2	�c6̩�A�*

epsilon�$)��ʅ�.       ��W�	x�d6̩�A�* 

Average reward per step�$)��.�       ��2	w�d6̩�A�*

epsilon�$)��V	�.       ��W�	x|h6̩�A�* 

Average reward per step�$)�`�5x       ��2	p}h6̩�A�*

epsilon�$)���.       ��W�	h^k6̩�A�* 

Average reward per step�$)���       ��2	x_k6̩�A�*

epsilon�$)�2h.       ��W�	3�m6̩�A�* 

Average reward per step�$)��M�       ��2	�m6̩�A�*

epsilon�$)��1.       ��W�	qsq6̩�A�* 

Average reward per step�$)��A�%       ��2	dtq6̩�A�*

epsilon�$)����.       ��W�	U3�6̩�A�* 

Average reward per step�$)��S        ��2	Y4�6̩�A�*

epsilon�$)�;k�.       ��W�	v��6̩�A�* 

Average reward per step�$)��:m       ��2	T��6̩�A�*

epsilon�$)�$�40       ���_	9�6̩�As*#
!
Average reward per episode]t��Ty+�.       ��W�	X:�6̩�As*!

total reward per episode  �i C�.       ��W�	e6�6̩�A�* 

Average reward per step]t��q�q       ��2	37�6̩�A�*

epsilon]t����6�.       ��W�	H��6̩�A�* 

Average reward per step]t��
O)&       ��2	Q��6̩�A�*

epsilon]t��^�S.       ��W�		7�6̩�A�* 

Average reward per step]t��y֛�       ��2	�9�6̩�A�*

epsilon]t���M��.       ��W�	���6̩�A�* 

Average reward per step]t����LT       ��2	���6̩�A�*

epsilon]t�����.       ��W�	]�6̩�A�* 

Average reward per step]t��w|       ��2	�]�6̩�A�*

epsilon]t��{��.       ��W�	��6̩�A�* 

Average reward per step]t������       ��2	��6̩�A�*

epsilon]t��*ib$.       ��W�	!�6̩�A�* 

Average reward per step]t����]       ��2	"�6̩�A�*

epsilon]t�����.       ��W�	�x�6̩�A�* 

Average reward per step]t��T�       ��2	�y�6̩�A�*

epsilon]t��k�pw.       ��W�	__�6̩�A�* 

Average reward per step]t����L�       ��2	�`�6̩�A�*

epsilon]t�����.       ��W�	���6̩�A�* 

Average reward per step]t��oM��       ��2	Y��6̩�A�*

epsilon]t��-���.       ��W�	���6̩�A�* 

Average reward per step]t��܎X�       ��2	l��6̩�A�*

epsilon]t�����@.       ��W�	��6̩�A�* 

Average reward per step]t��ç�       ��2	3�6̩�A�*

epsilon]t���zRm.       ��W�	٘�6̩�A�* 

Average reward per step]t��|���       ��2	ę�6̩�A�*

epsilon]t��v��.       ��W�	�6̩�A�* 

Average reward per step]t��<��       ��2	"�6̩�A�*

epsilon]t���lA.       ��W�	�\�6̩�A�* 

Average reward per step]t����)       ��2	}]�6̩�A�*

epsilon]t��}�s	.       ��W�	�"�6̩�A�* 

Average reward per step]t���ʰ       ��2	�#�6̩�A�*

epsilon]t���S��.       ��W�	�|�6̩�A�* 

Average reward per step]t��į�       ��2	 �6̩�A�*

epsilon]t���ߜ.       ��W�	���6̩�A�* 

Average reward per step]t��U���       ��2	���6̩�A�*

epsilon]t�����.       ��W�	�6̩�A�* 

Average reward per step]t���R�a       ��2	"�6̩�A�*

epsilon]t��.g/�.       ��W�	.8�6̩�A�* 

Average reward per step]t����        ��2	�8�6̩�A�*

epsilon]t�����'.       ��W�	��6̩�A�* 

Average reward per step]t��b��       ��2	���6̩�A�*

epsilon]t��)�.       ��W�	�|�6̩�A�* 

Average reward per step]t���	��       ��2	�}�6̩�A�*

epsilon]t���c��.       ��W�	��6̩�A�* 

Average reward per step]t��Ĥд       ��2	���6̩�A�*

epsilon]t�����.       ��W�	�'�6̩�A�* 

Average reward per step]t��LV��       ��2	p(�6̩�A�*

epsilon]t����|.       ��W�	� 7̩�A�* 

Average reward per step]t����O�       ��2	/� 7̩�A�*

epsilon]t�����.       ��W�	�g7̩�A�* 

Average reward per step]t���v�       ��2	�h7̩�A�*

epsilon]t��0��.       ��W�	�%7̩�A�* 

Average reward per step]t��v���       ��2	t&7̩�A�*

epsilon]t���=�w.       ��W�	f�7̩�A�* 

Average reward per step]t��k�Q       ��2	��7̩�A�*

epsilon]t��i�.       ��W�	�m7̩�A�* 

Average reward per step]t�����,       ��2	Dn7̩�A�*

epsilon]t��I��.       ��W�	;�7̩�A�* 

Average reward per step]t�����        ��2	�7̩�A�*

epsilon]t��i�I.       ��W�	�]7̩�A�* 

Average reward per step]t��1�K       ��2	t^7̩�A�*

epsilon]t���k.       ��W�	�7̩�A�* 

Average reward per step]t��7F}�       ��2	S�7̩�A�*

epsilon]t��u���.       ��W�	}�7̩�A�* 

Average reward per step]t��x@t       ��2	1�7̩�A�*

epsilon]t���g&�.       ��W�	>�7̩�A�* 

Average reward per step]t������       ��2	q�7̩�A�*

epsilon]t��koЗ.       ��W�	��7̩�A�* 

Average reward per step]t���D�       ��2	&�7̩�A�*

epsilon]t��{�t.       ��W�	�7̩�A�* 

Average reward per step]t��Ż!�       ��2	�7̩�A�*

epsilon]t��n/L�.       ��W�	>v 7̩�A�* 

Average reward per step]t������       ��2	!w 7̩�A�*

epsilon]t��\�<{.       ��W�	�V#7̩�A�* 

Average reward per step]t��!��       ��2	XW#7̩�A�*

epsilon]t�����~.       ��W�	q�$7̩�A�* 

Average reward per step]t��C���       ��2	h�$7̩�A�*

epsilon]t����Hu.       ��W�	�='7̩�A�* 

Average reward per step]t��mӶ�       ��2	�>'7̩�A�*

epsilon]t���*��.       ��W�	��)7̩�A�* 

Average reward per step]t���N       ��2	V�)7̩�A�*

epsilon]t����?�0       ���_	��)7̩�At*#
!
Average reward per episodeە(��}�n.       ��W�	y�)7̩�At*!

total reward per episode  ��Rv�k.       ��W�	�/27̩�A�* 

Average reward per stepە(�'^��       ��2	�027̩�A�*

epsilonە(���{b.       ��W�	��57̩�A�* 

Average reward per stepە(�<���       ��2	}�57̩�A�*

epsilonە(����8.       ��W�	7887̩�A�* 

Average reward per stepە(�wNƃ       ��2	*987̩�A�*

epsilonە(���.       ��W�	(�97̩�A�* 

Average reward per stepە(�'�#       ��2	�97̩�A�*

epsilonە(���z.       ��W�	5@<7̩�A�* 

Average reward per stepە(�+�       ��2	�@<7̩�A�*

