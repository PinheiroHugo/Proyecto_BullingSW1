<?php

namespace App\Http\Controllers\Web;

use App\Models\Evento;
use App\Models\Evidencia;
use Illuminate\Http\Request;

class EvidenciaControllerWeb extends Controller
{


    public function indexGuardia(Evento $evento)
    {
        $evidencias=Evidencia::where('evento_id',$evento->id)->get();
        
        return view('guardia.evidencia.index',compact('evidencias'));
    }

}
