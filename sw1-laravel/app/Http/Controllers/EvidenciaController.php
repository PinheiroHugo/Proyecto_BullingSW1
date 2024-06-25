<?php

namespace App\Http\Controllers;

use App\Models\Evidencia;
use App\Http\Requests\Evidencia\IndexEvidenciaRequest;
use App\Http\Requests\Evidencia\StoreEvidenciaRequest;
use App\Http\Requests\Evidencia\UpdateEvidenciaRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class EvidenciaController extends Controller
{

    public function index(IndexEvidenciaRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Evidencia> enviada correctamente';
        try {
            $responseArr['data'] = Evidencia::Evento($request->evento_id)
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


    public function store(StoreEvidenciaRequest $request)
    {
        $data = [
            'file'=>'',
            'evento_id'=>$request->evento_id,
            'tipo'=>$request->tipo
        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evidencia> Registro exitoso';

        try {
            DB::beginTransaction();

            /* if ($request->hasFile('file')) {
                $data['file'] = Storage::disk('public')->put('file', $request->file);
            } */

            //S3
            if ($request->hasFile('file')) {
                $data['file'] = Storage::disk('s3')->put('file', $request->file);
            }

            $evidencia = Evidencia::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Evidencia : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $evidencia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            /* if (!is_null($data['file'])) {
                Storage::disk('public')->delete($data['file']);
            } */

            //S3
            if (!is_null($data['file'])) {
                Storage::disk('s3')->delete($data['file']);
            }

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Evidencia $evidencia)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Evidencia> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $evidencia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Evidencia $evidencia)
    {
        //
    }

    public function update(UpdateEvidenciaRequest $request, Evidencia $evidencia)
    {
        $data = [
            'file'=>$evidencia->file,
            'evento_id'=>$request->evento_id,
            'tipo'=>$request->tipo
        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evidencia> Modificado correctamente';
        try {
            DB::beginTransaction();

            if ($request->hasFile('file')) {
                if (!is_null($evidencia->file)) {
                    Storage::disk('public')->delete($evidencia->file);
                }
                $data['file'] = Storage::disk('public')->put('file', $request->file);
            }


            $before = $evidencia;
            $evidencia->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Evidencia : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $evidencia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            if (!is_null($data['file'])) {
                Storage::disk('public')->delete($data['file']);
            }

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Evidencia $evidencia)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Evidencia> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$evidencia->id;
            if (!is_null($evidencia->file)) {
                Storage::disk('public')->delete($evidencia->file);
            }

            $evidencia->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Evidencia: con id:' . $id . '.'
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


    public function restore(Evidencia $evidencia)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Evidencia> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $evidencia->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Evidencia: ' . $evidencia->id
            //@@@     . ' con id:' . $evidencia->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $evidencia;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}
