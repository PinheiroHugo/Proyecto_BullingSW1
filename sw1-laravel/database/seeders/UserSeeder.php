<?php

namespace Database\Seeders;

use App\Models\Administrador;
use App\Models\Guardia;
use App\Models\Persona;
use App\Models\User;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;

class UserSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        $user1=User::create([
            'name'=>'User',
            'email'=>'user@example.com',
            'password'=>Hash::make('12345678')
        ]);

        $persona1=Persona::create([
            'nombre'=>'Laura',
            'apellido_p'=>'Ugarte',
            'apellido_m'=>'Arispe',
            'ci'=>'00000',
            'telefono'=>'62066339',
            'foto'=>'',
            'user_id'=>$user1->id,
        ]);

        $admin1=Administrador::create([
            'persona_id'=>$persona1->id
        ]);

        $guardia1=Guardia::create([
            'estado'=>'1',
            'fecha_ini'=>'2023/05/05',
            'persona_id'=>$persona1->id,
        ]);

    }
}
