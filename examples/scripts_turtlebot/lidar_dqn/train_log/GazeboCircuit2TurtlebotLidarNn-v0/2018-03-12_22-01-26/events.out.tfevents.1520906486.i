       £K"	  А=ћ©÷Abrain.Event:2єcФТЫё      ™*	@КЙ=ћ©÷A"Ољ
z
flatten_1_inputPlaceholder*
dtype0*+
_output_shapes
:€€€€€€€€€* 
shape:€€€€€€€€€
^
flatten_1/ShapeShapeflatten_1_input*
out_type0*
_output_shapes
:*
T0
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
flatten_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ѓ
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
€€€€€€€€€*
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
З
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
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
 *»~ўљ*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *»~ў=*
dtype0*
_output_shapes
: 
©
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	А*
seed2юЫѓ*
seed±€е)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
Н
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes
:	А

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	А
Д
dense_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	А*
	container *
shape:	А
љ
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	А
|
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes
:	А*
T0*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
dtype0*
_output_shapes	
:А*
valueBА*    
z
dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:А*
	container *
shape:А*
shared_name 
™
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:А
Щ
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
З
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*(
_output_shapes
:€€€€€€€€€А*
T0*
data_formatNHWC
]
activation_1/ReluReludense_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
m
dense_2/random_uniform/shapeConst*
_output_shapes
:*
valueB"   d   *
dtype0
_
dense_2/random_uniform/minConst*
valueB
 *?» љ*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *?» =*
dtype0*
_output_shapes
: 
©
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed±€е)*
T0*
dtype0*
_output_shapes
:	Аd*
seed2нъР
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
Н
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	Аd*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes
:	Аd*
T0
Д
dense_2/kernel
VariableV2*
dtype0*
_output_shapes
:	Аd*
	container *
shape:	Аd*
shared_name 
љ
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	Аd*
use_locking(*
T0
|
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Аd*
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
©
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
Ш
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d*
T0
\
activation_2/ReluReludense_2/BiasAdd*'
_output_shapes
:€€€€€€€€€d*
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
 *ЌћLЊ*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
®
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes

:d2*
seed2юЏБ*
seed±€е)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
T0*
_output_shapes
: 
М
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
_output_shapes

:d2*
T0
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:d2
В
dense_3/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:d2*
	container *
shape
:d2
Љ
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
validate_shape(*
_output_shapes

:d2
{
dense_3/kernel/readIdentitydense_3/kernel*
_output_shapes

:d2*
T0*!
_class
loc:@dense_3/kernel
Z
dense_3/ConstConst*
_output_shapes
:2*
valueB2*    *
dtype0
x
dense_3/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:2*
	container *
shape:2
©
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0*
_class
loc:@dense_3/bias
q
dense_3/bias/readIdentitydense_3/bias*
T0*
_class
loc:@dense_3/bias*
_output_shapes
:2
Ш
dense_3/MatMulMatMulactivation_2/Reludense_3/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
transpose_b( 
Ж
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€2*
T0
\
activation_3/ReluReludense_3/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€2
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
 *лDђЊ*
dtype0
_
dense_4/random_uniform/maxConst*
valueB
 *лDђ>*
dtype0*
_output_shapes
: 
І
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
T0*
dtype0*
_output_shapes

:2*
seed2№яK*
seed±€е)
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
М
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:2
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:2*
T0
В
dense_4/kernel
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
Љ
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
dense_4/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
x
dense_4/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
©
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
Ш
dense_4/MatMulMatMulactivation_3/Reludense_4/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Ж
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
d
activation_4/IdentityIdentitydense_4/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
m
dense_5/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
valueB
 *Мmњ*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Мm?
®
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2жнЛ*
seed±€е)
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
_output_shapes
: *
T0
М
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes

:
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
T0*
_output_shapes

:
В
dense_5/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
Љ
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
©
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
T0*
_class
loc:@dense_5/bias*
validate_shape(*
_output_shapes
:*
use_locking(
q
dense_5/bias/readIdentitydense_5/bias*
_output_shapes
:*
T0*
_class
loc:@dense_5/bias
Ц
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Ж
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
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
і
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:€€€€€€€€€
b
lambda_1/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
Р
lambda_1/ExpandDims
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
o
lambda_1/strided_slice_1/stackConst*
valueB"       *
dtype0*
_output_shapes
:
q
 lambda_1/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB"        *
dtype0
q
 lambda_1/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ј
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*
T0*'
_output_shapes
:€€€€€€€€€
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
 lambda_1/strided_slice_2/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
ј
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€*
Index0*
T0
_
lambda_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
Е
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
:€€€€€€€€€
_
Adam/iterations/initial_valueConst*
_output_shapes
: *
value	B	 R *
dtype0	
s
Adam/iterations
VariableV2*
dtype0	*
_output_shapes
: *
	container *
shape: *
shared_name 
Њ
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
Adam/lr/initial_valueConst*
valueB
 *oГ9*
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
Ю
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ѓ
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
 *wЊ?*
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
Ѓ
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
™
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
:€€€€€€€€€* 
shape:€€€€€€€€€*
dtype0
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
є
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
flatten_1_1/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Д
flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
^
flatten_1_1/stack/0Const*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
z
flatten_1_1/stackPackflatten_1_1/stack/0flatten_1_1/Prod*
_output_shapes
:*
T0*

axis *
N
Н
flatten_1_1/ReshapeReshapeflatten_1_input_1flatten_1_1/stack*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
o
dense_1_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_1_1/random_uniform/minConst*
valueB
 *»~ўљ*
dtype0*
_output_shapes
: 
a
dense_1_1/random_uniform/maxConst*
valueB
 *»~ў=*
dtype0*
_output_shapes
: 
ђ
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
dtype0*
_output_shapes
:	А*
seed2Сшh*
seed±€е)*
T0
А
dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
T0*
_output_shapes
: 
У
dense_1_1/random_uniform/mulMul&dense_1_1/random_uniform/RandomUniformdense_1_1/random_uniform/sub*
T0*
_output_shapes
:	А
Е
dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
T0*
_output_shapes
:	А
Ж
dense_1_1/kernel
VariableV2*
dtype0*
_output_shapes
:	А*
	container *
shape:	А*
shared_name 
≈
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	А
В
dense_1_1/kernel/readIdentitydense_1_1/kernel*
_output_shapes
:	А*
T0*#
_class
loc:@dense_1_1/kernel
^
dense_1_1/ConstConst*
dtype0*
_output_shapes	
:А*
valueBА*    
|
dense_1_1/bias
VariableV2*
dtype0*
_output_shapes	
:А*
	container *
shape:А*
shared_name 
≤
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
_output_shapes	
:А*
use_locking(*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(
x
dense_1_1/bias/readIdentitydense_1_1/bias*
T0*!
_class
loc:@dense_1_1/bias*
_output_shapes	
:А
Я
dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Н
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
a
activation_1_1/ReluReludense_1_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
o
dense_2_1/random_uniform/shapeConst*
valueB"   d   *
dtype0*
_output_shapes
:
a
dense_2_1/random_uniform/minConst*
valueB
 *?» љ*
dtype0*
_output_shapes
: 
a
dense_2_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?» =*
dtype0
≠
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	Аd*
seed2”ЉЉ*
seed±€е)
А
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
_output_shapes
: *
T0
У
dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
T0*
_output_shapes
:	Аd
Е
dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
_output_shapes
:	Аd*
T0
Ж
dense_2_1/kernel
VariableV2*
shape:	Аd*
shared_name *
dtype0*
_output_shapes
:	Аd*
	container 
≈
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	Аd
В
dense_2_1/kernel/readIdentitydense_2_1/kernel*
_output_shapes
:	Аd*
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:d*
	container *
shape:d
±
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
Ю
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
М
dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
`
activation_2_1/ReluReludense_2_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
o
dense_3_1/random_uniform/shapeConst*
valueB"d   2   *
dtype0*
_output_shapes
:
a
dense_3_1/random_uniform/minConst*
_output_shapes
: *
valueB
 *ЌћLЊ*
dtype0
a
dense_3_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЌћL>
ђ
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
_output_shapes

:d2*
seed2ыи¬*
seed±€е)*
T0*
dtype0
А
dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
T0*
_output_shapes
: 
Т
dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
T0*
_output_shapes

:d2
Д
dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
T0*
_output_shapes

:d2
Д
dense_3_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:d2*
	container *
shape
:d2
ƒ
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0*#
_class
loc:@dense_3_1/kernel
Б
dense_3_1/kernel/readIdentitydense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
_output_shapes

:d2*
T0
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
±
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2*
use_locking(*
T0
w
dense_3_1/bias/readIdentitydense_3_1/bias*
_output_shapes
:2*
T0*!
_class
loc:@dense_3_1/bias
Ю
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
transpose_b( 
М
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€2
`
activation_3_1/ReluReludense_3_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€2
o
dense_4_1/random_uniform/shapeConst*
valueB"2      *
dtype0*
_output_shapes
:
a
dense_4_1/random_uniform/minConst*
valueB
 *лDђЊ*
dtype0*
_output_shapes
: 
a
dense_4_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *лDђ>*
dtype0
Ђ
&dense_4_1/random_uniform/RandomUniformRandomUniformdense_4_1/random_uniform/shape*
_output_shapes

:2*
seed2÷єM*
seed±€е)*
T0*
dtype0
А
dense_4_1/random_uniform/subSubdense_4_1/random_uniform/maxdense_4_1/random_uniform/min*
_output_shapes
: *
T0
Т
dense_4_1/random_uniform/mulMul&dense_4_1/random_uniform/RandomUniformdense_4_1/random_uniform/sub*
T0*
_output_shapes

:2
Д
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
T0*
_output_shapes

:2
Д
dense_4_1/kernel
VariableV2*
dtype0*
_output_shapes

:2*
	container *
shape
:2*
shared_name 
ƒ
dense_4_1/kernel/AssignAssigndense_4_1/kerneldense_4_1/random_uniform*
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2*
use_locking(
Б
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
±
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
Ю
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
М
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
o
dense_5_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_5_1/random_uniform/minConst*
valueB
 *Мmњ*
dtype0*
_output_shapes
: 
a
dense_5_1/random_uniform/maxConst*
valueB
 *Мm?*
dtype0*
_output_shapes
: 
ђ
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2б≠…*
seed±€е)
А
dense_5_1/random_uniform/subSubdense_5_1/random_uniform/maxdense_5_1/random_uniform/min*
T0*
_output_shapes
: 
Т
dense_5_1/random_uniform/mulMul&dense_5_1/random_uniform/RandomUniformdense_5_1/random_uniform/sub*
T0*
_output_shapes

:
Д
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
T0*
_output_shapes

:
Д
dense_5_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
ƒ
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
use_locking(*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:
Б
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
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
±
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
Ь
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
М
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
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
Њ
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:€€€€€€€€€*
T0*
Index0
d
lambda_1_1/ExpandDims/dimConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
Ц
lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€*

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
"lambda_1_1/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB"      *
dtype0
 
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*
T0*'
_output_shapes
:€€€€€€€€€
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
 
lambda_1_1/strided_slice_2StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€*
Index0*
T0*
shrink_axis_mask 
a
lambda_1_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Л
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
:€€€€€€€€€
Ж
IsVariableInitializedIsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_output_shapes
: *
_class
loc:@dense_2/bias*
dtype0
И
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_4/kernel
Д
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_8IsVariableInitializeddense_5/kernel*!
_class
loc:@dense_5/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_9IsVariableInitializeddense_5/bias*
_output_shapes
: *
_class
loc:@dense_5/bias*
dtype0
Л
IsVariableInitialized_10IsVariableInitializedAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
{
IsVariableInitialized_11IsVariableInitializedAdam/lr*
_output_shapes
: *
_class
loc:@Adam/lr*
dtype0
Г
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
_output_shapes
: *
_class
loc:@Adam/beta_1*
dtype0
Г
IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
_output_shapes
: *
_class
loc:@Adam/beta_2*
dtype0
Б
IsVariableInitialized_14IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_15IsVariableInitializeddense_1_1/kernel*#
_class
loc:@dense_1_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_16IsVariableInitializeddense_1_1/bias*!
_class
loc:@dense_1_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_17IsVariableInitializeddense_2_1/kernel*#
_class
loc:@dense_2_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_18IsVariableInitializeddense_2_1/bias*!
_class
loc:@dense_2_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_19IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_20IsVariableInitializeddense_3_1/bias*!
_class
loc:@dense_3_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_21IsVariableInitializeddense_4_1/kernel*#
_class
loc:@dense_4_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_22IsVariableInitializeddense_4_1/bias*
_output_shapes
: *!
_class
loc:@dense_4_1/bias*
dtype0
Н
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*#
_class
loc:@dense_5_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*!
_class
loc:@dense_5_1/bias*
dtype0*
_output_shapes
: 
‘
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
shape:	А*
dtype0*
_output_shapes
:	А
І
AssignAssigndense_1_1/kernelPlaceholder*
use_locking( *
T0*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	А
X
Placeholder_1Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
£
Assign_1Assigndense_1_1/biasPlaceholder_1*
use_locking( *
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:А
`
Placeholder_2Placeholder*
dtype0*
_output_shapes
:	Аd*
shape:	Аd
Ђ
Assign_2Assigndense_2_1/kernelPlaceholder_2*
use_locking( *
T0*#
_class
loc:@dense_2_1/kernel*
validate_shape(*
_output_shapes
:	Аd
V
Placeholder_3Placeholder*
shape:d*
dtype0*
_output_shapes
:d
Ґ
Assign_3Assigndense_2_1/biasPlaceholder_3*
_output_shapes
:d*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(
^
Placeholder_4Placeholder*
dtype0*
_output_shapes

:d2*
shape
:d2
™
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
Ґ
Assign_5Assigndense_3_1/biasPlaceholder_5*
_output_shapes
:2*
use_locking( *
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(
^
Placeholder_6Placeholder*
shape
:2*
dtype0*
_output_shapes

:2
™
Assign_6Assigndense_4_1/kernelPlaceholder_6*
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(*
_output_shapes

:2*
use_locking( 
V
Placeholder_7Placeholder*
dtype0*
_output_shapes
:*
shape:
Ґ
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
_output_shapes

:*
shape
:*
dtype0
™
Assign_8Assigndense_5_1/kernelPlaceholder_8*
validate_shape(*
_output_shapes

:*
use_locking( *
T0*#
_class
loc:@dense_5_1/kernel
V
Placeholder_9Placeholder*
_output_shapes
:*
shape:*
dtype0
Ґ
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
Ї
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
„#<*
dtype0
j
SGD/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ъ
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
≤
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
SGD/decay/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
m
	SGD/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
¶
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
Д
lambda_1_targetPlaceholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
r
lambda_1_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*'
_output_shapes
:€€€€€€€€€*
T0
m
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*
T0*'
_output_shapes
:€€€€€€€€€
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
∞
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
≤
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:€€€€€€€€€

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*#
_output_shapes
:€€€€€€€€€*
T0
b
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
y
loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
b
loss/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Т
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ж
loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
d
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ч
loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
O

loss/mul/xConst*
valueB
 *  А?*
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
¬
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
„#<*
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
Ґ
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
a
SGD_1/lr/readIdentitySGD_1/lr*
_class
loc:@SGD_1/lr*
_output_shapes
: *
T0
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
Ї
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
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ѓ
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
Ж
lambda_1_target_1Placeholder*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€*
dtype0
t
lambda_1_sample_weights_1Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
r
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*
T0*'
_output_shapes
:€€€€€€€€€
q
loss_1/lambda_1_loss/SquareSquareloss_1/lambda_1_loss/sub*
T0*'
_output_shapes
:€€€€€€€€€
v
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
ґ
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
p
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Є
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
Е
loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*#
_output_shapes
:€€€€€€€€€*
T0
d
loss_1/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
У
loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*#
_output_shapes
:€€€€€€€€€*
T0
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
d
loss_1/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ш
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
М
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*#
_output_shapes
:€€€€€€€€€*
T0
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Э
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss_1/mul/xConst*
valueB
 *  А?*
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
shape:€€€€€€€€€*
dtype0*'
_output_shapes
:€€€€€€€€€
g
maskPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
Y

loss_2/subSublambda_1/suby_true*'
_output_shapes
:€€€€€€€€€*
T0
O

loss_2/AbsAbs
loss_2/sub*'
_output_shapes
:€€€€€€€€€*
T0
R
loss_2/Less/yConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
`
loss_2/LessLess
loss_2/Absloss_2/Less/y*'
_output_shapes
:€€€€€€€€€*
T0
U
loss_2/SquareSquare
loss_2/sub*
T0*'
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€
Q
loss_2/Abs_1Abs
loss_2/sub*
T0*'
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€
S
loss_2/mul_1/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*'
_output_shapes
:€€€€€€€€€*
T0
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*'
_output_shapes
:€€€€€€€€€*
T0
Z
loss_2/mul_2Mulloss_2/Selectmask*
T0*'
_output_shapes
:€€€€€€€€€
g
loss_2/Sum/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
И

loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:€€€€€€€€€
А
loss_targetPlaceholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
Ж
lambda_1_target_2Placeholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
n
loss_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
t
lambda_1_sample_weights_2Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Э
loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*#
_output_shapes
:€€€€€€€€€*
T0
`
loss_3/loss_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
u
loss_3/loss_loss/CastCastloss_3/loss_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
`
loss_3/loss_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
М
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
А
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*#
_output_shapes
:€€€€€€€€€*
T0
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
С
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Q
loss_3/mul/xConst*
valueB
 *  А?*
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
:€€€€€€€€€
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ї
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
Г
loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*
T0*#
_output_shapes
:€€€€€€€€€
d
loss_3/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
У
loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*#
_output_shapes
:€€€€€€€€€*
T0
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
d
loss_3/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ш
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
М
loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*
T0*#
_output_shapes
:€€€€€€€€€
f
loss_3/lambda_1_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
Э
loss_3/lambda_1_loss/Mean_2Meanloss_3/lambda_1_loss/truedivloss_3/lambda_1_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
S
loss_3/mul_1/xConst*
valueB
 *  А?*
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
:€€€€€€€€€
}
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*'
_output_shapes
:€€€€€€€€€*
T0

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ќ
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:€€€€€€€€€
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
≥
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ь
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
`
metrics_2/mean_q/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Й
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
О
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
Й
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
Е
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
Н
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
Й
IsVariableInitialized_31IsVariableInitializedSGD_1/momentum*!
_class
loc:@SGD_1/momentum*
dtype0*
_output_shapes
: 
Г
IsVariableInitialized_32IsVariableInitializedSGD_1/decay*
_class
loc:@SGD_1/decay*
dtype0*
_output_shapes
: 
Є
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign"…ІPF      “Иw	oфК=ћ©÷AJєА
ЏЇ
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
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
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
ref"dtypeА
is_initialized
"
dtypetypeШ
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
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Н
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
2	Р
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
Р
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
Н
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
2	И
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
ц
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
М
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.6.02v1.6.0-0-gd2e24b6039Ољ
z
flatten_1_inputPlaceholder*
dtype0*+
_output_shapes
:€€€€€€€€€* 
shape:€€€€€€€€€
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
ѓ
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
€€€€€€€€€*
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
З
flatten_1/ReshapeReshapeflatten_1_inputflatten_1/stack*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
m
dense_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *»~ўљ*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *»~ў=*
dtype0*
_output_shapes
: 
©
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
_output_shapes
:	А*
seed2юЫѓ*
seed±€е)*
T0*
dtype0
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
_output_shapes
: *
T0
Н
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes
:	А*
T0

dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes
:	А
Д
dense_1/kernel
VariableV2*
shape:	А*
shared_name *
dtype0*
_output_shapes
:	А*
	container 
љ
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
|
dense_1/kernel/readIdentitydense_1/kernel*
_output_shapes
:	А*
T0*!
_class
loc:@dense_1/kernel
\
dense_1/ConstConst*
valueBА*    *
dtype0*
_output_shapes	
:А
z
dense_1/bias
VariableV2*
dtype0*
_output_shapes	
:А*
	container *
shape:А*
shared_name 
™
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:А*
T0*
_class
loc:@dense_1/bias
Щ
dense_1/MatMulMatMulflatten_1/Reshapedense_1/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
З
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
]
activation_1/ReluReludense_1/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
m
dense_2/random_uniform/shapeConst*
valueB"   d   *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *?» љ*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?» =*
dtype0
©
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed±€е)*
T0*
dtype0*
_output_shapes
:	Аd*
seed2нъР
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
Н
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
_output_shapes
:	Аd*
T0

dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
T0*
_output_shapes
:	Аd
Д
dense_2/kernel
VariableV2*
shape:	Аd*
shared_name *
dtype0*
_output_shapes
:	Аd*
	container 
љ
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes
:	Аd
|
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes
:	Аd*
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
©
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
validate_shape(*
_output_shapes
:d*
use_locking(*
T0*
_class
loc:@dense_2/bias
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:d
Ш
dense_2/MatMulMatMulactivation_1/Reludense_2/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
Ж
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€d
\
activation_2/ReluReludense_2/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€d
m
dense_3/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"d   2   
_
dense_3/random_uniform/minConst*
valueB
 *ЌћLЊ*
dtype0*
_output_shapes
: 
_
dense_3/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ЌћL>
®
$dense_3/random_uniform/RandomUniformRandomUniformdense_3/random_uniform/shape*
dtype0*
_output_shapes

:d2*
seed2юЏБ*
seed±€е)*
T0
z
dense_3/random_uniform/subSubdense_3/random_uniform/maxdense_3/random_uniform/min*
_output_shapes
: *
T0
М
dense_3/random_uniform/mulMul$dense_3/random_uniform/RandomUniformdense_3/random_uniform/sub*
T0*
_output_shapes

:d2
~
dense_3/random_uniformAdddense_3/random_uniform/muldense_3/random_uniform/min*
T0*
_output_shapes

:d2
В
dense_3/kernel
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
Љ
dense_3/kernel/AssignAssigndense_3/kerneldense_3/random_uniform*
validate_shape(*
_output_shapes

:d2*
use_locking(*
T0*!
_class
loc:@dense_3/kernel
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
VariableV2*
_output_shapes
:2*
	container *
shape:2*
shared_name *
dtype0
©
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
Ш
dense_3/MatMulMatMulactivation_2/Reludense_3/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
transpose_b( 
Ж
dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€2
\
activation_3/ReluReludense_3/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€2
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
 *лDђЊ*
dtype0
_
dense_4/random_uniform/maxConst*
valueB
 *лDђ>*
dtype0*
_output_shapes
: 
І
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
_output_shapes

:2*
seed2№яK*
seed±€е)*
T0*
dtype0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0
М
dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
T0*
_output_shapes

:2
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:2*
T0
В
dense_4/kernel
VariableV2*
dtype0*
_output_shapes

:2*
	container *
shape
:2*
shared_name 
Љ
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(*
_output_shapes

:2
{
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel*
_output_shapes

:2*
T0
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
©
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
Ш
dense_4/MatMulMatMulactivation_3/Reludense_4/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Ж
dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
d
activation_4/IdentityIdentitydense_4/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
m
dense_5/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_5/random_uniform/minConst*
valueB
 *Мmњ*
dtype0*
_output_shapes
: 
_
dense_5/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *Мm?
®
$dense_5/random_uniform/RandomUniformRandomUniformdense_5/random_uniform/shape*
seed±€е)*
T0*
dtype0*
_output_shapes

:*
seed2жнЛ
z
dense_5/random_uniform/subSubdense_5/random_uniform/maxdense_5/random_uniform/min*
T0*
_output_shapes
: 
М
dense_5/random_uniform/mulMul$dense_5/random_uniform/RandomUniformdense_5/random_uniform/sub*
T0*
_output_shapes

:
~
dense_5/random_uniformAdddense_5/random_uniform/muldense_5/random_uniform/min*
_output_shapes

:*
T0
В
dense_5/kernel
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
Љ
dense_5/kernel/AssignAssigndense_5/kerneldense_5/random_uniform*
T0*!
_class
loc:@dense_5/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
{
dense_5/kernel/readIdentitydense_5/kernel*
_output_shapes

:*
T0*!
_class
loc:@dense_5/kernel
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
©
dense_5/bias/AssignAssigndense_5/biasdense_5/Const*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_5/bias*
validate_shape(
q
dense_5/bias/readIdentitydense_5/bias*
T0*
_class
loc:@dense_5/bias*
_output_shapes
:
Ц
dense_5/MatMulMatMuldense_4/BiasAdddense_5/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Ж
dense_5/BiasAddBiasAdddense_5/MatMuldense_5/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
m
lambda_1/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
o
lambda_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
o
lambda_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
і
lambda_1/strided_sliceStridedSlicedense_5/BiasAddlambda_1/strided_slice/stacklambda_1/strided_slice/stack_1lambda_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:€€€€€€€€€
b
lambda_1/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Р
lambda_1/ExpandDims
ExpandDimslambda_1/strided_slicelambda_1/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€*

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
ј
lambda_1/strided_slice_1StridedSlicedense_5/BiasAddlambda_1/strided_slice_1/stack lambda_1/strided_slice_1/stack_1 lambda_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€
t
lambda_1/addAddlambda_1/ExpandDimslambda_1/strided_slice_1*'
_output_shapes
:€€€€€€€€€*
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
ј
lambda_1/strided_slice_2StridedSlicedense_5/BiasAddlambda_1/strided_slice_2/stack lambda_1/strided_slice_2/stack_1 lambda_1/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€
_
lambda_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
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
:€€€€€€€€€
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
Њ
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
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
 *oГ9*
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
Ю
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
Ѓ
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
 *wЊ?*
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
Ѓ
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
T0
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
™
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
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
|
flatten_1_input_1Placeholder*
dtype0*+
_output_shapes
:€€€€€€€€€* 
shape:€€€€€€€€€
b
flatten_1_1/ShapeShapeflatten_1_input_1*
_output_shapes
:*
T0*
out_type0
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
є
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
Д
flatten_1_1/ProdProdflatten_1_1/strided_sliceflatten_1_1/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
^
flatten_1_1/stack/0Const*
valueB :
€€€€€€€€€*
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
Н
flatten_1_1/ReshapeReshapeflatten_1_input_1flatten_1_1/stack*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
o
dense_1_1/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
a
dense_1_1/random_uniform/minConst*
valueB
 *»~ўљ*
dtype0*
_output_shapes
: 
a
dense_1_1/random_uniform/maxConst*
valueB
 *»~ў=*
dtype0*
_output_shapes
: 
ђ
&dense_1_1/random_uniform/RandomUniformRandomUniformdense_1_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	А*
seed2Сшh*
seed±€е)
А
dense_1_1/random_uniform/subSubdense_1_1/random_uniform/maxdense_1_1/random_uniform/min*
T0*
_output_shapes
: 
У
dense_1_1/random_uniform/mulMul&dense_1_1/random_uniform/RandomUniformdense_1_1/random_uniform/sub*
T0*
_output_shapes
:	А
Е
dense_1_1/random_uniformAdddense_1_1/random_uniform/muldense_1_1/random_uniform/min*
T0*
_output_shapes
:	А
Ж
dense_1_1/kernel
VariableV2*
shape:	А*
shared_name *
dtype0*
_output_shapes
:	А*
	container 
≈
dense_1_1/kernel/AssignAssigndense_1_1/kerneldense_1_1/random_uniform*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking(*
T0
В
dense_1_1/kernel/readIdentitydense_1_1/kernel*
T0*#
_class
loc:@dense_1_1/kernel*
_output_shapes
:	А
^
dense_1_1/ConstConst*
dtype0*
_output_shapes	
:А*
valueBА*    
|
dense_1_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:А*
	container *
shape:А
≤
dense_1_1/bias/AssignAssigndense_1_1/biasdense_1_1/Const*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
x
dense_1_1/bias/readIdentitydense_1_1/bias*
T0*!
_class
loc:@dense_1_1/bias*
_output_shapes	
:А
Я
dense_1_1/MatMulMatMulflatten_1_1/Reshapedense_1_1/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Н
dense_1_1/BiasAddBiasAdddense_1_1/MatMuldense_1_1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
a
activation_1_1/ReluReludense_1_1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
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
 *?» љ*
dtype0
a
dense_2_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?» =*
dtype0
≠
&dense_2_1/random_uniform/RandomUniformRandomUniformdense_2_1/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	Аd*
seed2”ЉЉ*
seed±€е)
А
dense_2_1/random_uniform/subSubdense_2_1/random_uniform/maxdense_2_1/random_uniform/min*
T0*
_output_shapes
: 
У
dense_2_1/random_uniform/mulMul&dense_2_1/random_uniform/RandomUniformdense_2_1/random_uniform/sub*
_output_shapes
:	Аd*
T0
Е
dense_2_1/random_uniformAdddense_2_1/random_uniform/muldense_2_1/random_uniform/min*
T0*
_output_shapes
:	Аd
Ж
dense_2_1/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes
:	Аd*
	container *
shape:	Аd
≈
dense_2_1/kernel/AssignAssigndense_2_1/kerneldense_2_1/random_uniform*
validate_shape(*
_output_shapes
:	Аd*
use_locking(*
T0*#
_class
loc:@dense_2_1/kernel
В
dense_2_1/kernel/readIdentitydense_2_1/kernel*
T0*#
_class
loc:@dense_2_1/kernel*
_output_shapes
:	Аd
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
±
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
Ю
dense_2_1/MatMulMatMulactivation_1_1/Reludense_2_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
transpose_b( 
М
dense_2_1/BiasAddBiasAdddense_2_1/MatMuldense_2_1/bias/read*'
_output_shapes
:€€€€€€€€€d*
T0*
data_formatNHWC
`
activation_2_1/ReluReludense_2_1/BiasAdd*'
_output_shapes
:€€€€€€€€€d*
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
 *ЌћLЊ*
dtype0*
_output_shapes
: 
a
dense_3_1/random_uniform/maxConst*
valueB
 *ЌћL>*
dtype0*
_output_shapes
: 
ђ
&dense_3_1/random_uniform/RandomUniformRandomUniformdense_3_1/random_uniform/shape*
seed±€е)*
T0*
dtype0*
_output_shapes

:d2*
seed2ыи¬
А
dense_3_1/random_uniform/subSubdense_3_1/random_uniform/maxdense_3_1/random_uniform/min*
T0*
_output_shapes
: 
Т
dense_3_1/random_uniform/mulMul&dense_3_1/random_uniform/RandomUniformdense_3_1/random_uniform/sub*
T0*
_output_shapes

:d2
Д
dense_3_1/random_uniformAdddense_3_1/random_uniform/muldense_3_1/random_uniform/min*
_output_shapes

:d2*
T0
Д
dense_3_1/kernel
VariableV2*
dtype0*
_output_shapes

:d2*
	container *
shape
:d2*
shared_name 
ƒ
dense_3_1/kernel/AssignAssigndense_3_1/kerneldense_3_1/random_uniform*
_output_shapes

:d2*
use_locking(*
T0*#
_class
loc:@dense_3_1/kernel*
validate_shape(
Б
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
±
dense_3_1/bias/AssignAssigndense_3_1/biasdense_3_1/Const*
use_locking(*
T0*!
_class
loc:@dense_3_1/bias*
validate_shape(*
_output_shapes
:2
w
dense_3_1/bias/readIdentitydense_3_1/bias*
T0*!
_class
loc:@dense_3_1/bias*
_output_shapes
:2
Ю
dense_3_1/MatMulMatMulactivation_2_1/Reludense_3_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
transpose_b( 
М
dense_3_1/BiasAddBiasAdddense_3_1/MatMuldense_3_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€2
`
activation_3_1/ReluReludense_3_1/BiasAdd*'
_output_shapes
:€€€€€€€€€2*
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
 *лDђЊ*
dtype0
a
dense_4_1/random_uniform/maxConst*
_output_shapes
: *
valueB
 *лDђ>*
dtype0
Ђ
&dense_4_1/random_uniform/RandomUniformRandomUniformdense_4_1/random_uniform/shape*
dtype0*
_output_shapes

:2*
seed2÷єM*
seed±€е)*
T0
А
dense_4_1/random_uniform/subSubdense_4_1/random_uniform/maxdense_4_1/random_uniform/min*
T0*
_output_shapes
: 
Т
dense_4_1/random_uniform/mulMul&dense_4_1/random_uniform/RandomUniformdense_4_1/random_uniform/sub*
T0*
_output_shapes

:2
Д
dense_4_1/random_uniformAdddense_4_1/random_uniform/muldense_4_1/random_uniform/min*
_output_shapes

:2*
T0
Д
dense_4_1/kernel
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
shared_name *
dtype0
ƒ
dense_4_1/kernel/AssignAssigndense_4_1/kerneldense_4_1/random_uniform*
_output_shapes

:2*
use_locking(*
T0*#
_class
loc:@dense_4_1/kernel*
validate_shape(
Б
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
VariableV2*
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
±
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
Ю
dense_4_1/MatMulMatMulactivation_3_1/Reludense_4_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
М
dense_4_1/BiasAddBiasAdddense_4_1/MatMuldense_4_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
o
dense_5_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
dense_5_1/random_uniform/minConst*
valueB
 *Мmњ*
dtype0*
_output_shapes
: 
a
dense_5_1/random_uniform/maxConst*
valueB
 *Мm?*
dtype0*
_output_shapes
: 
ђ
&dense_5_1/random_uniform/RandomUniformRandomUniformdense_5_1/random_uniform/shape*
seed±€е)*
T0*
dtype0*
_output_shapes

:*
seed2б≠…
А
dense_5_1/random_uniform/subSubdense_5_1/random_uniform/maxdense_5_1/random_uniform/min*
T0*
_output_shapes
: 
Т
dense_5_1/random_uniform/mulMul&dense_5_1/random_uniform/RandomUniformdense_5_1/random_uniform/sub*
T0*
_output_shapes

:
Д
dense_5_1/random_uniformAdddense_5_1/random_uniform/muldense_5_1/random_uniform/min*
T0*
_output_shapes

:
Д
dense_5_1/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
ƒ
dense_5_1/kernel/AssignAssigndense_5_1/kerneldense_5_1/random_uniform*
T0*#
_class
loc:@dense_5_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
Б
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
±
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
Ь
dense_5_1/MatMulMatMuldense_4_1/BiasAdddense_5_1/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
М
dense_5_1/BiasAddBiasAdddense_5_1/MatMuldense_5_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
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
Њ
lambda_1_1/strided_sliceStridedSlicedense_5_1/BiasAddlambda_1_1/strided_slice/stack lambda_1_1/strided_slice/stack_1 lambda_1_1/strided_slice/stack_2*
end_mask*#
_output_shapes
:€€€€€€€€€*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
d
lambda_1_1/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
Ц
lambda_1_1/ExpandDims
ExpandDimslambda_1_1/strided_slicelambda_1_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
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
 
lambda_1_1/strided_slice_1StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_1/stack"lambda_1_1/strided_slice_1/stack_1"lambda_1_1/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:€€€€€€€€€*
Index0*
T0
z
lambda_1_1/addAddlambda_1_1/ExpandDimslambda_1_1/strided_slice_1*
T0*'
_output_shapes
:€€€€€€€€€
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
 
lambda_1_1/strided_slice_2StridedSlicedense_5_1/BiasAdd lambda_1_1/strided_slice_2/stack"lambda_1_1/strided_slice_2/stack_1"lambda_1_1/strided_slice_2/stack_2*'
_output_shapes
:€€€€€€€€€*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
a
lambda_1_1/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
Л
lambda_1_1/MeanMeanlambda_1_1/strided_slice_2lambda_1_1/Const*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:
h
lambda_1_1/subSublambda_1_1/addlambda_1_1/Mean*'
_output_shapes
:€€€€€€€€€*
T0
Ж
IsVariableInitializedIsVariableInitializeddense_1/kernel*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
dtype0
Д
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_4IsVariableInitializeddense_3/kernel*!
_class
loc:@dense_3/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_5IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_6IsVariableInitializeddense_4/kernel*!
_class
loc:@dense_4/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_7IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_8IsVariableInitializeddense_5/kernel*
_output_shapes
: *!
_class
loc:@dense_5/kernel*
dtype0
Д
IsVariableInitialized_9IsVariableInitializeddense_5/bias*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
: 
Л
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
Г
IsVariableInitialized_12IsVariableInitializedAdam/beta_1*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1
Г
IsVariableInitialized_13IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 
Б
IsVariableInitialized_14IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_15IsVariableInitializeddense_1_1/kernel*#
_class
loc:@dense_1_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_16IsVariableInitializeddense_1_1/bias*!
_class
loc:@dense_1_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_17IsVariableInitializeddense_2_1/kernel*#
_class
loc:@dense_2_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_18IsVariableInitializeddense_2_1/bias*!
_class
loc:@dense_2_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_19IsVariableInitializeddense_3_1/kernel*#
_class
loc:@dense_3_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_20IsVariableInitializeddense_3_1/bias*!
_class
loc:@dense_3_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_21IsVariableInitializeddense_4_1/kernel*#
_class
loc:@dense_4_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_22IsVariableInitializeddense_4_1/bias*!
_class
loc:@dense_4_1/bias*
dtype0*
_output_shapes
: 
Н
IsVariableInitialized_23IsVariableInitializeddense_5_1/kernel*#
_class
loc:@dense_5_1/kernel*
dtype0*
_output_shapes
: 
Й
IsVariableInitialized_24IsVariableInitializeddense_5_1/bias*!
_class
loc:@dense_5_1/bias*
dtype0*
_output_shapes
: 
‘
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^dense_1_1/kernel/Assign^dense_1_1/bias/Assign^dense_2_1/kernel/Assign^dense_2_1/bias/Assign^dense_3_1/kernel/Assign^dense_3_1/bias/Assign^dense_4_1/kernel/Assign^dense_4_1/bias/Assign^dense_5_1/kernel/Assign^dense_5_1/bias/Assign
^
PlaceholderPlaceholder*
shape:	А*
dtype0*
_output_shapes
:	А
І
AssignAssigndense_1_1/kernelPlaceholder*#
_class
loc:@dense_1_1/kernel*
validate_shape(*
_output_shapes
:	А*
use_locking( *
T0
X
Placeholder_1Placeholder*
shape:А*
dtype0*
_output_shapes	
:А
£
Assign_1Assigndense_1_1/biasPlaceholder_1*
T0*!
_class
loc:@dense_1_1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking( 
`
Placeholder_2Placeholder*
shape:	Аd*
dtype0*
_output_shapes
:	Аd
Ђ
Assign_2Assigndense_2_1/kernelPlaceholder_2*
validate_shape(*
_output_shapes
:	Аd*
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
Ґ
Assign_3Assigndense_2_1/biasPlaceholder_3*
_output_shapes
:d*
use_locking( *
T0*!
_class
loc:@dense_2_1/bias*
validate_shape(
^
Placeholder_4Placeholder*
dtype0*
_output_shapes

:d2*
shape
:d2
™
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
Ґ
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
™
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
Ґ
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
™
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
Ґ
Assign_9Assigndense_5_1/biasPlaceholder_9*
use_locking( *
T0*!
_class
loc:@dense_5_1/bias*
validate_shape(*
_output_shapes
:
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
Ї
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
SGD/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *
„#<
j
SGD/lr
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ъ
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/lr
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
≤
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(
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
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
¶
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
Д
lambda_1_targetPlaceholder*%
shape:€€€€€€€€€€€€€€€€€€*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
r
lambda_1_sample_weightsPlaceholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
p
loss/lambda_1_loss/subSublambda_1_1/sublambda_1_target*
T0*'
_output_shapes
:€€€€€€€€€
m
loss/lambda_1_loss/SquareSquareloss/lambda_1_loss/sub*'
_output_shapes
:€€€€€€€€€*
T0
t
)loss/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
∞
loss/lambda_1_loss/MeanMeanloss/lambda_1_loss/Square)loss/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
n
+loss/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
≤
loss/lambda_1_loss/Mean_1Meanloss/lambda_1_loss/Mean+loss/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 

loss/lambda_1_loss/mulMulloss/lambda_1_loss/Mean_1lambda_1_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
b
loss/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Н
loss/lambda_1_loss/NotEqualNotEquallambda_1_sample_weightsloss/lambda_1_loss/NotEqual/y*#
_output_shapes
:€€€€€€€€€*
T0
y
loss/lambda_1_loss/CastCastloss/lambda_1_loss/NotEqual*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

b
loss/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Т
loss/lambda_1_loss/Mean_2Meanloss/lambda_1_loss/Castloss/lambda_1_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ж
loss/lambda_1_loss/truedivRealDivloss/lambda_1_loss/mulloss/lambda_1_loss/Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
d
loss/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ч
loss/lambda_1_loss/Mean_3Meanloss/lambda_1_loss/truedivloss/lambda_1_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
O

loss/mul/xConst*
valueB
 *  А?*
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
¬
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
„#<*
dtype0*
_output_shapes
: 
l
SGD_1/lr
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ґ
SGD_1/lr/AssignAssignSGD_1/lrSGD_1/lr/initial_value*
_class
loc:@SGD_1/lr*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
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
Ї
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
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
Ѓ
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
Ж
lambda_1_target_1Placeholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
t
lambda_1_sample_weights_1Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
r
loss_1/lambda_1_loss/subSublambda_1/sublambda_1_target_1*'
_output_shapes
:€€€€€€€€€*
T0
q
loss_1/lambda_1_loss/SquareSquareloss_1/lambda_1_loss/sub*
T0*'
_output_shapes
:€€€€€€€€€
v
+loss_1/lambda_1_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ґ
loss_1/lambda_1_loss/MeanMeanloss_1/lambda_1_loss/Square+loss_1/lambda_1_loss/Mean/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 
p
-loss_1/lambda_1_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
Є
loss_1/lambda_1_loss/Mean_1Meanloss_1/lambda_1_loss/Mean-loss_1/lambda_1_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 
Е
loss_1/lambda_1_loss/mulMulloss_1/lambda_1_loss/Mean_1lambda_1_sample_weights_1*
T0*#
_output_shapes
:€€€€€€€€€
d
loss_1/lambda_1_loss/NotEqual/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
У
loss_1/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_1loss_1/lambda_1_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
}
loss_1/lambda_1_loss/CastCastloss_1/lambda_1_loss/NotEqual*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
d
loss_1/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ш
loss_1/lambda_1_loss/Mean_2Meanloss_1/lambda_1_loss/Castloss_1/lambda_1_loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
М
loss_1/lambda_1_loss/truedivRealDivloss_1/lambda_1_loss/mulloss_1/lambda_1_loss/Mean_2*#
_output_shapes
:€€€€€€€€€*
T0
f
loss_1/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Э
loss_1/lambda_1_loss/Mean_3Meanloss_1/lambda_1_loss/truedivloss_1/lambda_1_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Q
loss_1/mul/xConst*
valueB
 *  А?*
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
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
g
maskPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
Y

loss_2/subSublambda_1/suby_true*
T0*'
_output_shapes
:€€€€€€€€€
O

loss_2/AbsAbs
loss_2/sub*'
_output_shapes
:€€€€€€€€€*
T0
R
loss_2/Less/yConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
`
loss_2/LessLess
loss_2/Absloss_2/Less/y*
T0*'
_output_shapes
:€€€€€€€€€
U
loss_2/SquareSquare
loss_2/sub*
T0*'
_output_shapes
:€€€€€€€€€
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
:€€€€€€€€€
Q
loss_2/Abs_1Abs
loss_2/sub*'
_output_shapes
:€€€€€€€€€*
T0
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
:€€€€€€€€€
S
loss_2/mul_1/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0
c
loss_2/mul_1Mulloss_2/mul_1/xloss_2/sub_1*'
_output_shapes
:€€€€€€€€€*
T0
p
loss_2/SelectSelectloss_2/Less
loss_2/mulloss_2/mul_1*'
_output_shapes
:€€€€€€€€€*
T0
Z
loss_2/mul_2Mulloss_2/Selectmask*'
_output_shapes
:€€€€€€€€€*
T0
g
loss_2/Sum/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
И

loss_2/SumSumloss_2/mul_2loss_2/Sum/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 
А
loss_targetPlaceholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
Ж
lambda_1_target_2Placeholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
n
loss_sample_weightsPlaceholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
t
lambda_1_sample_weights_2Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
j
'loss_3/loss_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
Э
loss_3/loss_loss/MeanMean
loss_2/Sum'loss_3/loss_loss/Mean/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( 
u
loss_3/loss_loss/mulMulloss_3/loss_loss/Meanloss_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
`
loss_3/loss_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
loss_3/loss_loss/NotEqualNotEqualloss_sample_weightsloss_3/loss_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
u
loss_3/loss_loss/CastCastloss_3/loss_loss/NotEqual*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

`
loss_3/loss_loss/ConstConst*
_output_shapes
:*
valueB: *
dtype0
М
loss_3/loss_loss/Mean_1Meanloss_3/loss_loss/Castloss_3/loss_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
А
loss_3/loss_loss/truedivRealDivloss_3/loss_loss/mulloss_3/loss_loss/Mean_1*
T0*#
_output_shapes
:€€€€€€€€€
b
loss_3/loss_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
С
loss_3/loss_loss/Mean_2Meanloss_3/loss_loss/truedivloss_3/loss_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Q
loss_3/mul/xConst*
valueB
 *  А?*
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
:€€€€€€€€€
u
+loss_3/lambda_1_loss/Mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ї
loss_3/lambda_1_loss/MeanMeanloss_3/lambda_1_loss/zeros_like+loss_3/lambda_1_loss/Mean/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
Г
loss_3/lambda_1_loss/mulMulloss_3/lambda_1_loss/Meanlambda_1_sample_weights_2*
T0*#
_output_shapes
:€€€€€€€€€
d
loss_3/lambda_1_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
У
loss_3/lambda_1_loss/NotEqualNotEquallambda_1_sample_weights_2loss_3/lambda_1_loss/NotEqual/y*#
_output_shapes
:€€€€€€€€€*
T0
}
loss_3/lambda_1_loss/CastCastloss_3/lambda_1_loss/NotEqual*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

d
loss_3/lambda_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ш
loss_3/lambda_1_loss/Mean_1Meanloss_3/lambda_1_loss/Castloss_3/lambda_1_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
М
loss_3/lambda_1_loss/truedivRealDivloss_3/lambda_1_loss/mulloss_3/lambda_1_loss/Mean_1*#
_output_shapes
:€€€€€€€€€*
T0
f
loss_3/lambda_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Э
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
 *  А?*
dtype0
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
:€€€€€€€€€
}
!metrics_2/mean_absolute_error/AbsAbs!metrics_2/mean_absolute_error/sub*'
_output_shapes
:€€€€€€€€€*
T0

4metrics_2/mean_absolute_error/Mean/reduction_indicesConst*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
ќ
"metrics_2/mean_absolute_error/MeanMean!metrics_2/mean_absolute_error/Abs4metrics_2/mean_absolute_error/Mean/reduction_indices*#
_output_shapes
:€€€€€€€€€*

Tidx0*
	keep_dims( *
T0
m
#metrics_2/mean_absolute_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
≥
$metrics_2/mean_absolute_error/Mean_1Mean"metrics_2/mean_absolute_error/Mean#metrics_2/mean_absolute_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
q
&metrics_2/mean_q/Max/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ь
metrics_2/mean_q/MaxMaxlambda_1/sub&metrics_2/mean_q/Max/reduction_indices*

Tidx0*
	keep_dims( *
T0*#
_output_shapes
:€€€€€€€€€
`
metrics_2/mean_q/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Й
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
О
metrics_2/mean_q/Mean_1Meanmetrics_2/mean_q/Meanmetrics_2/mean_q/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Й
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
Е
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
Н
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
Й
IsVariableInitialized_31IsVariableInitializedSGD_1/momentum*!
_class
loc:@SGD_1/momentum*
dtype0*
_output_shapes
: 
Г
IsVariableInitialized_32IsVariableInitializedSGD_1/decay*
_class
loc:@SGD_1/decay*
dtype0*
_output_shapes
: 
Є
init_1NoOp^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^SGD/decay/Assign^SGD_1/iterations/Assign^SGD_1/lr/Assign^SGD_1/momentum/Assign^SGD_1/decay/Assign""з
trainable_variablesѕћ
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
SGD_1/decay:0SGD_1/decay/AssignSGD_1/decay/read:02SGD_1/decay/initial_value:0"Ё
	variablesѕћ
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
