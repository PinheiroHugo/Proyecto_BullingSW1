<?php

namespace Database\Seeders;

use App\Models\Camara;
use Illuminate\Database\Seeder;

class CamaraSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        Camara::create([
            'codigo'=>'C001',
            'ubicacion'=>'Principal',
    
        ]);
    }
}
