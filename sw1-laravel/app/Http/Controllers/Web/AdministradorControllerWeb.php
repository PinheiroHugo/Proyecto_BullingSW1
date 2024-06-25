<?php

namespace App\Http\Controllers\Web;

use App\Models\Administrador;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class AdministradorControllerWeb extends Controller
{


    public function loginView()
    {
        return view('admin.login');
    }


    public function login(Request $request)
    {
        $credentials = $this->validate(
            request(),
            [
                'email' => 'required|string',
                'password' => 'required|string'
            ]
        );

        $user = Administrador::join('personas', 'personas.id', 'administradors.persona_id')->join('users', 'users.id', 'personas.user_id')
            ->where('email', $request->email)->select('administradors.*', 'users.password')->get()->first();

        if (!is_null($user)) {
            if (Hash::check($request->password, $user->password)) {
                auth()->guard('admin')->login($user, false);
                return redirect()->route('admin.dashboard');
            }
        }


        return back()
            ->withErrors(['email' => 'Estas credenciales no concuerdan con nuestros registros.'])
            ->withInput(request(['email']));
    }

    public function logout()
    {
        auth()->guard('admin')->logout();
        return redirect()->route('admin.login.view');
    }



    public function dashboardView()
    {
        return view('admin.dashboard');
    }

}
