<?php

namespace App\Http\Controllers;

use App\Models\Persona;
use App\Http\Requests\Persona\IndexPersonaRequest;
use App\Http\Requests\Persona\StorePersonaRequest;
use App\Http\Requests\Persona\UpdatePersonaRequest;
use Illuminate\Http\Response;
use Illuminate\Support\Facades\Storage;
use \Exception;
use Illuminate\Support\Facades\DB;

class PersonaController extends Controller
{

    public function index(IndexPersonaRequest $request)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = 'Lista<Persona> enviada correctamente';
        try {
            $responseArr['data'] = Persona::User($request->user_id)
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


    public function store(StorePersonaRequest $request)
    {
        $data = [
            'nombre'=>$request->nombre,
            'apellido_p'=>$request->apellido_p,
            'apellido_m'=>$request->apellido_m,
            'ci'=>$request->ci,
            'telefono'=>$request->telefono,
            'foto'=>'',
            'user_id'=>$request->user_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Persona> Registro exitoso';

        try {
            DB::beginTransaction();

            /* if ($request->hasFile('foto')) {
                $data['foto'] = Storage::disk('public')->put('foto', $request->foto);
            } */

            //S3
            if ($request->hasFile('foto')) {
                $data['foto'] = Storage::disk('s3')->put('foto', $request->foto);
            }

            $persona = Persona::create($data);

            //@@@ $description = '';
            //@@@ foreach ($data as $d) {
            //@@@     $index = array_search($d, $data);
            //@@@     $description = $description .   $index . " : " . $d . ",\n";
            //@@@ };
            //@@@
            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' creo Persona : \n'.
            //@@@     '[ '.$description.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $persona;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();

            if (!is_null($data['foto'])) {
                Storage::disk('public')->delete($data['foto']);
            }

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }


    public function show(Persona $persona)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Persona> ¡Enviado correctamente!';
        try {
            $responseArr['data'] = $persona;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function edit(Persona $persona)
    {
        //
    }

    public function update(UpdatePersonaRequest $request, Persona $persona)
    {
        $data = [
            'nombre'=>$request->nombre,
            'apellido_p'=>$request->apellido_p,
            'apellido_m'=>$request->apellido_m,
            'ci'=>$request->ci,
            'telefono'=>$request->telefono,
            'foto'=>$persona->foto,
            'user_id'=>$request->user_id,

        ];

        $responseArr['data'] = [];
        $responseArr['message'] = '<Persona> Modificado correctamente';
        try {
            DB::beginTransaction();

            if ($request->hasFile('foto')) {
                if (!is_null($persona->foto)) {
                    Storage::disk('public')->delete($persona->foto);
                }
                $data['foto'] = Storage::disk('public')->put('foto', $request->foto);
            }


            $before = $persona;
            $persona->update($data);

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
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' actualizo Persona : \n'.
            //@@@     '[ '.$description.'] \n con:  ['.$description_before.' ]'
            //@@@ ]);

            DB::commit();

            $responseArr['data'] = $persona;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            if (!is_null($data['foto'])) {
                Storage::disk('public')->delete($data['foto']);
            }

            $message = $e;
            $responseArr['data'] = [];
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }

    public function destroy(Persona $persona)
    {
        $responseArr['data'] = [];
        $responseArr['message'] = '<Persona> Eliminado correctamente';
        try {
            DB::beginTransaction();

            $id=$persona->id;
            if (!is_null($persona->foto)) {
                Storage::disk('public')->delete($persona->foto);
            }

            $persona->delete();


            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' eliminó Persona: con id:' . $id . '.'
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


    public function restore(Persona $persona)
    {

        $responseArr['data'] = [];
        $responseArr['message'] = '<Persona> Restaurado correctamente';
        try {
            DB::beginTransaction();

            $persona->restore();

            //@@@ Binnacle::create([
            //@@@     'user_id' => auth()->user()->id,
            //@@@     'description' => 'El usuario ' . auth()->user()->name . ' restauró Persona: ' . $persona->id
            //@@@     . ' con id:' . $persona->id . '.'
            //@@@ ]);
            DB::commit();

            $responseArr['data'] = $persona;
            return response()->json($responseArr, Response::HTTP_OK);
        } catch (Exception $e) {
            DB::rollBack();
            $message = $e;
            $responseArr['message'] = $message;
            return response()->json($responseArr, Response::HTTP_GATEWAY_TIMEOUT);
        }
    }
}
