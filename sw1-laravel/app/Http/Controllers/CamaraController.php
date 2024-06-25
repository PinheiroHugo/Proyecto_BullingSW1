<?php

namespace App\Http\Controllers;

use App\Models\Camara;
use App\Http\Requests\Camara\IndexCamaraRequest;
use App\Http\Requests\Camara\StoreCamaraRequest;
use App\Http\Requests\Camara\UpdateCamaraRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class CamaraController extends Controller
{

    public function index(IndexCamaraRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Camara> enviada correctamente';
        try {
            $responseArr['data'] = Camara::all();
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


    public function store(StoreCamaraRequest $request)
    {
        $data = [
            'codigo'=>$request->codigo,
            'ubicacion'=>$request->ubicacion,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Camara> Registro exitoso';

        try {
            DB::beginTransaction();


            $camara = Camara::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Camara : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $camara;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();


            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Camara $camara)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Camara> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $camara;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Camara $camara)
    {
        //
    }

    public function update(UpdateCamaraRequest $request, Camara $camara)
    {
        $data = [
            'codigo'=>$request->codigo,
            'ubicacion'=>$request->ubicacion,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Camara> Modificado correctamente';
        try {
            DB::beginTransaction();



            $before = $camara;
            $camara->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Camara : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $camara;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Camara $camara)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Camara> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$camara->id;

            $camara->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Camara: con id:' . $id . '.'
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


    public function restore(Camara $camara)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Camara> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $camara->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Camara: ' . $camara->id 
            //@@@     . ' con id:' . $camara->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $camara;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}