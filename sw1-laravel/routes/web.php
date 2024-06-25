<?php

use App\Http\Controllers\Web\GuardiaControllerWeb;
use App\Http\Controllers\Web\AdministradorControllerWeb;
use App\Http\Controllers\Web\EventoControllerWeb;
use App\Http\Controllers\Web\EvidenciaControllerWeb;
use App\Http\Controllers\Web\InformeControllerWeb;
use App\Http\Controllers\Web\PageControllerWeb;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

Route::get('/', [PageControllerWeb::class, 'welcome'])->name('welcome');


Route::prefix('admin')->group(function () {

    /* Dos login con diferentes roles (admin y guardia) */
    Route::middleware(['guest:admin', 'guest:guardia'])->group(function () {
        Route::get('/login', [AdministradorControllerWeb::class, 'loginView'])->name('admin.login.view');
        Route::post('/login', [AdministradorControllerWeb::class, 'login'])->name('admin.login');
    });

    Route::middleware(['auth:admin'])->group(function () {
        Route::get('/dashboard', [AdministradorControllerWeb::class, 'dashboardView'])->name('admin.dashboard');
        Route::post('/logout',[AdministradorControllerWeb::class,'logout'])->name('admin.logout');

        Route::get('/informes', [InformeControllerWeb::class, 'indexAdmin'])->name('admin.informes.index');

    });
});


/* Route::prefix('guardia')->group(function () { */

    Route::middleware(['guest:admin', 'guest:guardia'])->group(function () {
        Route::get('/login', [GuardiaControllerWeb::class, 'loginView'])->name('guardia.login.view');
        Route::post('/login', [GuardiaControllerWeb::class, 'login'])->name('guardia.login');
    });

    Route::middleware(['auth:guardia'])->group(function () {
        Route::get('/dashboard', [GuardiaControllerWeb::class, 'dashboardView'])->name('guardia.dashboard');
        Route::post('/logout',[GuardiaControllerWeb::class,'logout'])->name('guardia.logout');


        Route::get('/informes', [InformeControllerWeb::class, 'indexGuardia'])->name('guardia.informes.index');
        Route::get('/eventos', [EventoControllerWeb::class, 'indexGuardia'])->name('guardia.eventos.index');
        Route::get('/crear_informe/{evento}', [InformeControllerWeb::class, 'createView'])->name('guardia.informe.crear');
        Route::post('/crear_informe/{evento}', [InformeControllerWeb::class, 'store'])->name('guardia.informe.store');
        Route::get('/evidencias/{evento}', [EvidenciaControllerWeb::class, 'indexGuardia'])->name('guardia.evidencia.index');

    });
/* }); */
