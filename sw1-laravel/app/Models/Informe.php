<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Informe extends Model
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'titulo',
        'documento',
        'guardia_id',
        'evento_id',

    ];

    public function scopeGuardia($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('guardia_id',$id); }
    }
    public function scopeEvento($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('evento_id',$id); }
    }

}
