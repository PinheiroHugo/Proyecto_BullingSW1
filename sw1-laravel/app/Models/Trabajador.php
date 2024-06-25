<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Trabajador extends Model
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'codigo_emp',
        'estado',
        'persona_id',

    ];

    public function scopePersona($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('persona_id',$id); }
    }

}
