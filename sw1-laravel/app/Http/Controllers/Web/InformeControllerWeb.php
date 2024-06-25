<?php

namespace App\Http\Controllers\Web;

use App\Models\Evento;
use App\Models\Guardia;
use App\Models\Informe;
use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;

class InformeControllerWeb extends Controller
{

    public function indexAdmin()
    {
        $informes=Informe::all();

        return view('admin.informe.index',compact('informes'));
    }

    public function indexGuardia()
    {
        $informes=Informe::all();

        return view('guardia.informe.index',compact('informes'));
    }

    public function createView(Evento $evento){

        return view('guardia.informe.crear',compact('evento'));
    }

    public function store(Request $request){

        $credentials = $this->validate(
            request(),
            [
                'evento_id' => 'required|string|exists:eventos,id',
                'titulo' => 'required|string',
                'documento' => 'required',
            ]
        );


        $data=[
            'evento_id'=>$request->evento_id,
            'titulo'=>$request->titulo,
            'guardia_id'=>auth()->user()->id,
            'documento'=>'',
        ];

        /* if ($request->hasFile('documento')) {
            $data['documento'] = Storage::disk('public')->put('documentos', $request->documento);
        } */

        //S3
        if ($request->hasFile('documento')) {
            $data['documento'] = Storage::disk('s3')->put('documentos', $request->documento);
        }

        Informe::create($data);

        return redirect()->route('guardia.informes.index',$request->evento_id);

    }

}
