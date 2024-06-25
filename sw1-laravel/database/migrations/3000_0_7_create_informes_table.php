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
        Schema::create('informes', function (Blueprint $table) { 
            $table->id();
            $table->string('titulo');
            $table->string('documento');
            $table->unsignedBigInteger('guardia_id');
            $table->foreign('guardia_id')->references('id')->on('guardias')->onDelete('cascade')->onUpdate('cascade');
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
        Schema::dropIfExists('informes');
    }
};
