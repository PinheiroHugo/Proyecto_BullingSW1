<?php

namespace App\Http\Controllers;

use App\Models\Trabajador;
use App\Http\Requests\Trabajador\IndexTrabajadorRequest;
use App\Http\Requests\Trabajador\StoreTrabajadorRequest;
use App\Http\Requests\Trabajador\UpdateTrabajadorRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class TrabajadorController extends Controller
{

    public function index(IndexTrabajadorRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Trabajador> enviada correctamente';
        try {
            $responseArr['data'] = Trabajador::Persona($request->persona_id)
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


    public function store(StoreTrabajadorRequest $request)
    {
        $data = [
            'codigo_emp'=>$request->codigo_emp,
            'estado'=>$request->estado,
            'persona_id'=>$request->persona_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Trabajador> Registro exitoso';

        try {
            DB::beginTransaction();


            $trabajador = Trabajador::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Trabajador : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $trabajador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Trabajador $trabajador)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Trabajador> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $trabajador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Trabajador $trabajador)
    {
        //
    }

    public function update(UpdateTrabajadorRequest $request, Trabajador $trabajador)
    {
        $data = [
            'codigo_emp'=>$request->codigo_emp,
            'estado'=>$request->estado,
            'persona_id'=>$request->persona_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Trabajador> Modificado correctamente';
        try {
            DB::beginTransaction();



            $before = $trabajador;
            $trabajador->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Trabajador : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $trabajador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Trabajador $trabajador)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Trabajador> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$trabajador->id;

            $trabajador->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Trabajador: con id:' . $id . '.'
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


    public function restore(Trabajador $trabajador)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Trabajador> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $trabajador->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Trabajador: ' . $trabajador->id 
            //@@@     . ' con id:' . $trabajador->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $trabajador;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}