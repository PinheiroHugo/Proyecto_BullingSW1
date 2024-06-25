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
        Schema::create('eventos', function (Blueprint $table) { 
            $table->id();
            $table->date('fecha');
            $table->string('descripcion');
            $table->boolean('es_queja');
            $table->unsignedBigInteger('trabajador_id')->nullable();
            $table->foreign('trabajador_id')->references('id')->on('trabajadors')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedBigInteger('camara_id');
            $table->foreign('camara_id')->references('id')->on('camaras')->onDelete('cascade')->onUpdate('cascade');
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
        Schema::dropIfExists('eventos');
    }
};
