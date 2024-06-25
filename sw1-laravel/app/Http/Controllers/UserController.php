<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Http\Requests\User\LoginUserRequest;
use App\Http\Requests\User\IndexUserRequest;
use App\Http\Requests\User\StoreUserRequest;
use App\Http\Requests\User\UpdateUserRequest;
use \Exception;
use Illuminate\Support\Facades\DB;
use Illuminate\Http\Request;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Hash;

class UserController extends Controller
{
    public function login(LoginUserRequest $request)
    {
        $user = User::where('email', $request->email)->get()->first();

        if (Hash::check($request->password, $user->password)) {

            $responseArr['data'] = $user;
            $responseArr['message'] = '¡El usuario se ha logeado correctamente!';
            $responseArr['token'] = $user->createToken('user', ['administrator'])->plainTextToken;
            return response()->json($responseArr, Response::HTTP_OK);
        }

        $message = '¡La contraseña es incorrecta!';

        $responseArr['data'] = [];
        $responseArr['message'] = $message;
        $responseArr['token'] = '';
        return response()->json($responseArr, Response::HTTP_UNAUTHORIZED);
    }

    public function logout()
    {
        auth()->user()->currentAccessToken()->delete();
        $responseArr = array();
        $responseArr['data'] = [];
        $responseArr['message'] = '¡El usuario ha cerrado sesion correctamente!';
        return response()->json($responseArr, Response::HTTP_OK);
    }


    public function logoutAll()
    {
        auth()->user()->tokens()->delete();
        $responseArr = array();
        $responseArr['data'] = [];
        $responseArr['message'] = '¡El usuario ha cerrado sesion correctamente en todas sus cuentas!';
        return response()->json($responseArr, Response::HTTP_OK);
    }

    public function signup(Request $request){
        $validator = Validator::make($request->all(), [
            'name' => 'required|string',
            'email' => 'required|email|string|unique:users,email',
            'password' => 'required|min:8|same:password_confirm',
            'password_confirm'=>'required'

        ]);

        $responseArr=array();
        if ($validator->fails()) {
            $responseArr['status']=false;
            $responseArr['data']=[];
            $responseArr['message'] = $validator->messages()->first();
            $responseArr['is_valid']=0;
            $responseArr['token'] = '';
            return response()->json($responseArr,Response::HTTP_BAD_REQUEST);
        }

        $data=[
            'name'=>$request->name,
            'email'=>$request->email,
            'password'=>Hash::make($request->password)
        ];

        $user=User::create($data);

        $responseArr['status']=true;
        $responseArr['data']=$user;
        $responseArr['message'] = '¡El usuario se ha registrado correctamente!';
        $responseArr['is_valid']=1;
        $responseArr['token'] = $user->createToken('myapptoken')->plainTextToken;
        return response()->json($responseArr,Response::HTTP_OK);
    }


    //////////////////////////////////////////////////////////////////////////////////
    public function index(IndexUserRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<User> enviada correctamente';
        try {
            $responseArr['data'] = User::all();
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function create()
    {
        //
    }


    public function store(StoreUserRequest $request)
    {
        $data = [
            'name'=>$request->name,
            'email'=>$request->email,
            'password'=>Hash::make($request->password)
        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<User> Registro exitoso';

        try {
            DB::beginTransaction();

            $user = User::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo User : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $user;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(User $user)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<User> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $user;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(User $user)
    {
        //
    }

    public function update(UpdateUserRequest $request, User $user)
    {
        $data = [
            'name'=>$request->name,
            'email'=>$request->email,
        ];

        if (!is_null($request->password) && $request->password!=''){
            $data['password']=Hash::make($request->password);
        }

        $responseArr['data'] = [];
        $responseArr['message'] = '<User> Modificado correctamente';
        try {
            DB::beginTransaction();

            $before = $user;
            $user->update($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ $description_before = '';
            //@@@ foreach ($before as $b) {
            //@@@     $index = array_search($b, $before);
            //@@@     $description_before = $description_before .   $index . " : " . $b . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo User : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $user;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(User $user)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<User> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$user->id;
            $user->delete();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó User: con id:' . $id . '.'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = [];
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function restore(User $user)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<User> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $user->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró User: ' . $user->id
            //@@@     . ' con id:' . $user->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $user;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

}
