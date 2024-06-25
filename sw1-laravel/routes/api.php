<?php

use App\Http\Controllers\UserController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

use App\Http\Controllers\PersonaController;
use App\Http\Controllers\AdministradorController;
use App\Http\Controllers\GuardiaController;
use App\Http\Controllers\TrabajadorController;
use App\Http\Controllers\CamaraController;
use App\Http\Controllers\EventoController;
use App\Http\Controllers\EvidenciaController;
use App\Http\Controllers\InformeController;
use App\Http\Controllers\TrabajadorEventoController;


/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| is assigned the "api" middleware group. Enjoy building your API!
|
*/

Route::post('/login',[UserController::class,'login']);
Route::post('/signup',[UserController::class,'signup']);

Route::middleware(['auth:sanctum'])->group(function () {
    Route::post('/user/authenticated', [UserController::class, 'authenticated']);
    Route::post('/logout', [UserController::class, 'logout']);
    Route::post('/logout_all', [UserController::class, 'logoutAll']);
    
    Route::prefix('user')->group(function () {
        Route::get('/',[UserController::class,'index']);
        Route::get('/{user}',[UserController::class,'show']);
        Route::post('/',[UserController::class,'store']);
        Route::put('/{user}',[UserController::class,'update']);
        Route::delete('/{user}',[UserController::class,'destroy']);
        Route::post('/restore',[UserController::class,'restore']);
    });

    Route::prefix('persona')->group(function () {
        Route::get('/',[PersonaController::class,'index']);
        Route::get('/{persona}',[PersonaController::class,'show']);
        Route::post('/',[PersonaController::class,'store']);
        Route::put('/{persona}',[PersonaController::class,'update']);
        Route::delete('/{persona}',[PersonaController::class,'destroy']);
        Route::post('/restore',[PersonaController::class,'restore']);
    });

    Route::prefix('administrador')->group(function () {
        Route::get('/',[AdministradorController::class,'index']);
        Route::get('/{administrador}',[AdministradorController::class,'show']);
        Route::post('/',[AdministradorController::class,'store']);
        Route::put('/{administrador}',[AdministradorController::class,'update']);
        Route::delete('/{administrador}',[AdministradorController::class,'destroy']);
        Route::post('/restore',[AdministradorController::class,'restore']);
    });

    Route::prefix('guardia')->group(function () {
        Route::get('/',[GuardiaController::class,'index']);
        Route::get('/{guardia}',[GuardiaController::class,'show']);
        Route::post('/',[GuardiaController::class,'store']);
        Route::put('/{guardia}',[GuardiaController::class,'update']);
        Route::delete('/{guardia}',[GuardiaController::class,'destroy']);
        Route::post('/restore',[GuardiaController::class,'restore']);
    });

    Route::prefix('trabajador')->group(function () {
        Route::get('/',[TrabajadorController::class,'index']);
        Route::get('/{trabajador}',[TrabajadorController::class,'show']);
        Route::post('/',[TrabajadorController::class,'store']);
        Route::put('/{trabajador}',[TrabajadorController::class,'update']);
        Route::delete('/{trabajador}',[TrabajadorController::class,'destroy']);
        Route::post('/restore',[TrabajadorController::class,'restore']);
    });

    Route::prefix('camara')->group(function () {
        Route::get('/',[CamaraController::class,'index']);
        Route::get('/{camara}',[CamaraController::class,'show']);
        Route::post('/',[CamaraController::class,'store']);
        Route::put('/{camara}',[CamaraController::class,'update']);
        Route::delete('/{camara}',[CamaraController::class,'destroy']);
        Route::post('/restore',[CamaraController::class,'restore']);
    });

    Route::prefix('evento')->group(function () {
        Route::get('/',[EventoController::class,'index']);
        Route::get('/{evento}',[EventoController::class,'show']);
        // Route::post('/',[EventoController::class,'store']);
        Route::put('/{evento}',[EventoController::class,'update']);
        Route::delete('/{evento}',[EventoController::class,'destroy']);
        Route::post('/restore',[EventoController::class,'restore']);
    });

    Route::prefix('evidencia')->group(function () {
        Route::get('/',[EvidenciaController::class,'index']);
        Route::get('/{evidencia}',[EvidenciaController::class,'show']);
        // Route::post('/',[EvidenciaController::class,'store']);
        Route::put('/{evidencia}',[EvidenciaController::class,'update']);
        Route::delete('/{evidencia}',[EvidenciaController::class,'destroy']);
        Route::post('/restore',[EvidenciaController::class,'restore']);
    });

    Route::prefix('informe')->group(function () {
        Route::get('/',[InformeController::class,'index']);
        Route::get('/{informe}',[InformeController::class,'show']);
        Route::post('/',[InformeController::class,'store']);
        Route::put('/{informe}',[InformeController::class,'update']);
        Route::delete('/{informe}',[InformeController::class,'destroy']);
        Route::post('/restore',[InformeController::class,'restore']);
    });

    Route::prefix('trabajadorevento')->group(function () {
        Route::get('/',[TrabajadorEventoController::class,'index']);
        Route::get('/{trabajadorevento}',[TrabajadorEventoController::class,'show']);
        Route::post('/',[TrabajadorEventoController::class,'store']);
        Route::put('/{trabajadorevento}',[TrabajadorEventoController::class,'update']);
        Route::delete('/{trabajadorevento}',[TrabajadorEventoController::class,'destroy']);
        Route::post('/restore',[TrabajadorEventoController::class,'restore']);
    });



    //Muestra de como usar las habilities
    Route::post('/test1', function () {
        return 'works';
    })->middleware('ability:user_store,administrator');
});



Route::prefix('evento')->group(function () {
    Route::post('/',[EventoController::class,'store']);
    Route::post('/evidencia',[EventoController::class,'storeEvidencia']);
});

Route::prefix('evidencia')->group(function () {
    Route::post('/',[EvidenciaController::class,'store']);
});
