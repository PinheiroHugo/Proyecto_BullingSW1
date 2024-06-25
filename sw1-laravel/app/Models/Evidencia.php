<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;

class Evidencia extends Model
{
    use HasFactory,SoftDeletes;

    protected $fillable=[
        'file',
        'evento_id',
        'tipo'
    ];

    public function scopeEvento($query,$id){
      if (is_null($id)) { return $query; }else{ return $query->where('evento_id',$id); }
    }

}
