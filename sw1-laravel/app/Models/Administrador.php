<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Database\Eloquent\SoftDeletes;

class Administrador extends Authenticatable
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'persona_id',

    ];

    public function scopePersona($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('persona_id',$id); }
    }

}
