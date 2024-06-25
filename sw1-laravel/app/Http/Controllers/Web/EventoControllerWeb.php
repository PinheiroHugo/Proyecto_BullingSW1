<?php

namespace App\Http\Controllers\Web;

use App\Models\Evento;
use Illuminate\Http\Request;

class EventoControllerWeb extends Controller
{


    public function indexGuardia()
    {
        $eventos=Evento::join('camaras','camaras.id','eventos.camara_id')
        ->select('eventos.*','camaras.codigo','camaras.ubicacion')->get();
        
        return view('guardia.evento.index',compact('eventos'));
    }

}
