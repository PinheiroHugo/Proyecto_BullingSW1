<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('trabajador_eventos', function (Blueprint $table) { 
            $table->id();
            $table->date('fecha');
            $table->string('descripcion');
            $table->unsignedBigInteger('trabajador_id');
            $table->foreign('trabajador_id')->references('id')->on('trabajadors')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedBigInteger('evento_id');
            $table->foreign('evento_id')->references('id')->on('eventos')->onDelete('cascade')->onUpdate('cascade');
            $table->timestamps();
            $table->softDeletes();
        });

    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('trabajador_eventos');
    }
};
