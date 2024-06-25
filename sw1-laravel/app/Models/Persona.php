<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Persona extends Model
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'nombre',
        'apellido_p',
        'apellido_m',
        'ci',
        'telefono',
        'foto',
        'user_id',

    ];

    public function scopeUser($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('user_id',$id); }
    }

}
