<?php

namespace App\Http\Controllers;

use App\Models\Informe;
use App\Http\Requests\Informe\IndexInformeRequest;
use App\Http\Requests\Informe\StoreInformeRequest;
use App\Http\Requests\Informe\UpdateInformeRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class InformeController extends Controller
{

    public function index(IndexInformeRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Informe> enviada correctamente';
        try {
            $responseArr['data'] = Informe::Guardia($request->guardia_id)
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


    public function store(StoreInformeRequest $request)
    {
        $data = [
            'titulo'=>$request->titulo,
            'documento'=>'',
            'guardia_id'=>$request->guardia_id,
            'evento_id'=>$request->evento_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Informe> Registro exitoso';

        try {
            DB::beginTransaction();

            /* if ($request->hasFile('documento')) {
                $data['documento'] = Storage::disk('public')->put('documento', $request->documento);
            } */

            //S3
            if ($request->hasFile('documento')) {
                $data['documento'] = Storage::disk('s3')->put('documento', $request->documento);
            }

            $informe = Informe::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Informe : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $informe;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            if (!is_null($data['documento'])) {
                Storage::disk('public')->delete($data['documento']);
            }

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Informe $informe)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Informe> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $informe;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Informe $informe)
    {
        //
    }

    public function update(UpdateInformeRequest $request, Informe $informe)
    {
        $data = [
            'titulo'=>$request->titulo,
            'documento'=>$informe->documento,
            'guardia_id'=>$request->guardia_id,
            'evento_id'=>$request->evento_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Informe> Modificado correctamente';
        try {
            DB::beginTransaction();

            if ($request->hasFile('documento')) {
                if (!is_null($informe->documento)) {
                    Storage::disk('public')->delete($informe->documento);
                }
                $data['documento'] = Storage::disk('public')->put('documento', $request->documento);
            }


            $before = $informe;
            $informe->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Informe : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $informe;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            if (!is_null($data['documento'])) {
                Storage::disk('public')->delete($data['documento']);
            }

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Informe $informe)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Informe> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$informe->id;
            if (!is_null($informe->documento)) {
                Storage::disk('public')->delete($informe->documento);
            }

            $informe->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Informe: con id:' . $id . '.'
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


    public function restore(Informe $informe)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Informe> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $informe->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Informe: ' . $informe->id
            //@@@     . ' con id:' . $informe->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $informe;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}
