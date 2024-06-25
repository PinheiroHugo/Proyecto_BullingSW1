<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Evento extends Model
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'fecha',
        'descripcion',
        'es_queja',
        'trabajador_id',
        'camara_id',

    ];

    public function scopeTrabajador($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('trabajador_id',$id); }
    }
    public function scopeCamara($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('camara_id',$id); }
    }

}
