<?php

namespace App\Http\Controllers;

use App\Models\Guardia;
use App\Http\Requests\Guardia\IndexGuardiaRequest;
use App\Http\Requests\Guardia\StoreGuardiaRequest;
use App\Http\Requests\Guardia\UpdateGuardiaRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class GuardiaController extends Controller
{

    public function index(IndexGuardiaRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Guardia> enviada correctamente';
        try {
            $responseArr['data'] = Guardia::Persona($request->persona_id)
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


    public function store(StoreGuardiaRequest $request)
    {
        $data = [
            'estado'=>$request->estado,
            'fecha_ini'=>$request->fecha_ini,
            'fecha_fin'=>$request->fecha_fin,
            'persona_id'=>$request->persona_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Guardia> Registro exitoso';

        try {
            DB::beginTransaction();


            $guardia = Guardia::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Guardia : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $guardia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Guardia $guardia)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Guardia> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $guardia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Guardia $guardia)
    {
        //
    }

    public function update(UpdateGuardiaRequest $request, Guardia $guardia)
    {
        $data = [
            'estado'=>$request->estado,
            'fecha_ini'=>$request->fecha_ini,
            'fecha_fin'=>$request->fecha_fin,
            'persona_id'=>$request->persona_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Guardia> Modificado correctamente';
        try {
            DB::beginTransaction();



            $before = $guardia;
            $guardia->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Guardia : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $guardia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Guardia $guardia)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Guardia> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$guardia->id;

            $guardia->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Guardia: con id:' . $id . '.'
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


    public function restore(Guardia $guardia)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Guardia> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $guardia->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Guardia: ' . $guardia->id 
            //@@@     . ' con id:' . $guardia->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $guardia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}