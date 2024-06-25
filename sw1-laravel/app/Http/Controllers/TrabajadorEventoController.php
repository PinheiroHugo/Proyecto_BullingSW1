<?php

namespace App\Http\Controllers;

use App\Models\TrabajadorEvento;
use App\Http\Requests\TrabajadorEvento\IndexTrabajadorEventoRequest;
use App\Http\Requests\TrabajadorEvento\StoreTrabajadorEventoRequest;
use App\Http\Requests\TrabajadorEvento\UpdateTrabajadorEventoRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class TrabajadorEventoController extends Controller
{

    public function index(IndexTrabajadorEventoRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<TrabajadorEvento> enviada correctamente';
        try {
            $responseArr['data'] = TrabajadorEvento::Trabajador($request->trabajador_id)
                                   ->Evento($request->evento_id)
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


    public function store(StoreTrabajadorEventoRequest $request)
    {
        $data = [
            'fecha'=>$request->fecha,
            'descripcion'=>$request->descripcion,
            'trabajador_id'=>$request->trabajador_id,
            'evento_id'=>$request->evento_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<TrabajadorEvento> Registro exitoso';

        try {
            DB::beginTransaction();


            $trabajadorevento = TrabajadorEvento::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo TrabajadorEvento : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $trabajadorevento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(TrabajadorEvento $trabajadorevento)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<TrabajadorEvento> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $trabajadorevento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(TrabajadorEvento $trabajadorevento)
    {
        //
    }

    public function update(UpdateTrabajadorEventoRequest $request, TrabajadorEvento $trabajadorevento)
    {
        $data = [
            'fecha'=>$request->fecha,
            'descripcion'=>$request->descripcion,
            'trabajador_id'=>$request->trabajador_id,
            'evento_id'=>$request->evento_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<TrabajadorEvento> Modificado correctamente';
        try {
            DB::beginTransaction();



            $before = $trabajadorevento;
            $trabajadorevento->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo TrabajadorEvento : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $trabajadorevento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(TrabajadorEvento $trabajadorevento)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<TrabajadorEvento> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$trabajadorevento->id;

            $trabajadorevento->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó TrabajadorEvento: con id:' . $id . '.'
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


    public function restore(TrabajadorEvento $trabajadorevento)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<TrabajadorEvento> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $trabajadorevento->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró TrabajadorEvento: ' . $trabajadorevento->id 
            //@@@     . ' con id:' . $trabajadorevento->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $trabajadorevento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}