<?php

namespace App\Http\Controllers\Web;


class PageControllerWeb extends Controller
{


    public function welcome()
    {
        return view('welcome');
    }

}
