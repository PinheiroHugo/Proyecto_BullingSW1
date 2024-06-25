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
        Schema::create('guardias', function (Blueprint $table) { 
            $table->id();
            $table->boolean('estado');
            $table->date('fecha_ini');
            $table->date('fecha_fin')->nullable();
            $table->unsignedBigInteger('persona_id');
            $table->foreign('persona_id')->references('id')->on('personas')->onDelete('cascade')->onUpdate('cascade');
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
        Schema::dropIfExists('guardias');
    }
};
