<?php

namespace App\Http\Controllers\Web;

use App\Models\Guardia;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class GuardiaControllerWeb extends Controller
{


    public function loginView()
    {
        return view('guardia.login');
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

        $user = Guardia::join('personas', 'personas.id', 'guardias.persona_id')->join('users', 'users.id', 'personas.user_id')
            ->where('email', $request->email)->select('guardias.*', 'users.password')->get()->first();

        if (!is_null($user)) {
            if (Hash::check($request->password, $user->password)) {

                auth()->guard('guardia')->login($user, false);
                return redirect()->route('guardia.dashboard');
            }
        }


        return back()
            ->withErrors(['email' => 'Estas credenciales no concuerdan con nuestros registros.'])
            ->withInput(request(['email']));
    }

    public function logout()
    {
        auth()->guard('guardia')->logout();
        return redirect()->route('guardia.login.view');
    }

    public function dashboardView()
    {
        return view('guardia.dashboard');
    }
}
