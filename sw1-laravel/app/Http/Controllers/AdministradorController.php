<?php

namespace App\Http\Controllers;

use App\Models\Administrador;
use App\Http\Requests\Administrador\IndexAdministradorRequest;
use App\Http\Requests\Administrador\StoreAdministradorRequest;
use App\Http\Requests\Administrador\UpdateAdministradorRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class AdministradorController extends Controller
{

    public function index(IndexAdministradorRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Administrador> enviada correctamente';
        try {
            $responseArr['data'] = Administrador::Persona($request->persona_id)
                                   ->get();
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


    public function store(StoreAdministradorRequest $request)
    {
        $data = [
            'persona_id'=>$request->persona_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Administrador> Registro exitoso';

        try {
            DB::beginTransaction();


            $administrador = Administrador::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Administrador : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $administrador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Administrador $administrador)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Administrador> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $administrador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Administrador $administrador)
    {
        //
    }

    public function update(UpdateAdministradorRequest $request, Administrador $administrador)
    {
        $data = [
            'persona_id'=>$request->persona_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Administrador> Modificado correctamente';
        try {
            DB::beginTransaction();



            $before = $administrador;
            $administrador->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Administrador : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $administrador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Administrador $administrador)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Administrador> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$administrador->id;

            $administrador->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Administrador: con id:' . $id . '.'
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


    public function restore(Administrador $administrador)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Administrador> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $administrador->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Administrador: ' . $administrador->id 
            //@@@     . ' con id:' . $administrador->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $administrador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}