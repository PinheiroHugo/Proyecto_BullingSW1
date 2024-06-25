<?php

namespace App\Http\Controllers;

use App\Models\Evento;
use App\Http\Requests\Evento\IndexEventoRequest;
use App\Http\Requests\Evento\StoreEventoRequest;
use App\Http\Requests\Evento\UpdateEventoRequest;
use App\Models\Evidencia;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class EventoController extends Controller
{

    public function index(IndexEventoRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Evento> enviada correctamente';
        try {
            $responseArr['data'] = Evento::Trabajador($request->trabajador_id)
                ->Camara($request->camara_id)
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


    public function store(StoreEventoRequest $request)
    {
        $data = [
            'fecha' => $request->fecha,
            'descripcion' => $request->descripcion,
            'es_queja' => $request->es_queja,
            'trabajador_id' => $request->trabajador_id,
            'camara_id' => $request->camara_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evento> Registro exitoso';

        try {
            DB::beginTransaction();


            $evento = Evento::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Evento : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $evento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function storeEvidencia(StoreEventoRequest $request)
    {
        $data = [
            'fecha' => $request->fecha,
            'descripcion' => $request->descripcion,
            'es_queja' => $request->es_queja,
            'trabajador_id' => $request->trabajador_id,
            'camara_id' => $request->camara_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evento> Registro exitoso';

        try {
            DB::beginTransaction();


            $evento = Evento::create($data);


            $data2 = [
                'file' => '',
                'evento_id' => $evento->id,
                'tipo' => $request->tipo
            ];

            //S3
            if ($request->hasFile('file')) {
                $data2['file'] = Storage::disk('s3')->put('file', $request->file);
            }

            $evidencia = Evidencia::create($data2);

            DB::commit();

            $responseArr['data'] = $evento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function show(Evento $evento)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Evento> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $evento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Evento $evento)
    {
        //
    }

    public function update(UpdateEventoRequest $request, Evento $evento)
    {
        $data = [
            'fecha' => $request->fecha,
            'descripcion' => $request->descripcion,
            'es_queja' => $request->es_queja,
            'trabajador_id' => $request->trabajador_id,
            'camara_id' => $request->camara_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evento> Modificado correctamente';
        try {
            DB::beginTransaction();



            $before = $evento;
            $evento->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Evento : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $evento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Evento $evento)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Evento> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id = $evento->id;

            $evento->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Evento: con id:' . $id . '.'
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


    public function restore(Evento $evento)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evento> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $evento->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Evento: ' . $evento->id
            //@@@     . ' con id:' . $evento->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $evento;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}
