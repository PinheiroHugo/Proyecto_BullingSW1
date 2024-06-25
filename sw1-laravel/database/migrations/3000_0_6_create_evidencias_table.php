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
        Schema::create('evidencias', function (Blueprint $table) { 
            $table->id();
            $table->string('file');
            $table->unsignedBigInteger('evento_id');
            $table->foreign('evento_id')->references('id')->on('eventos')->onDelete('cascade')->onUpdate('cascade');
            $table->string('tipo');
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
        Schema::dropIfExists('evidencias');
    }
};
