<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class TrabajadorEvento extends Model
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'fecha',
        'descripcion',
        'trabajador_id',
        'evento_id',

    ];

    public function scopeTrabajador($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('trabajador_id',$id); }
    }
    public function scopeEvento($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('evento_id',$id); }
    }

}
