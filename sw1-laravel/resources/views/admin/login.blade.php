<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <meta name="description" content="" />
    <meta name="author" content="" />
    <title>Login Admin</title>
    <link href="{{ asset('template/css/styles.css') }}" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.11.2/js/all.min.js" crossorigin="anonymous"></script>
</head>

<body class="bg-primary">
    <div id="layoutAuthentication">
        <div id="layoutAuthentication_content">
            <main>
                <div class="container">
                    <div class="row justify-content-center">
                        <div class="col-lg-5">
                            <div class="card shadow-lg border-0 rounded-lg mt-5">
                                <div class="card-header">
                                    <h3 class="text-center font-weight-light my-4">Iniciar Sesion como Administrador
                                    </h3>
                                </div>
                                <div class="card-body">
                                    <form action="{{ route('admin.login') }}" method="POST">
                                        {{ csrf_field() }}

                                        @if ($errors->any())
                                            @foreach ($errors->all() as $error)
                                                <div class="alert alert-danger">{{ $error }}</div>
                                            @endforeach
                                        @endif

                                        <div class="form-group"><label class="small mb-1"
                                                for="inputEmailAddress">Email</label>
                                            <input class="form-control py-4" id="inputEmailAddress" type="email"
                                                placeholder="Ingresa tu correo" name="email" />
                                        </div>
                                        <div class="form-group"><label class="small mb-1"
                                                for="inputPassword">Contraseña</label>
                                            <input class="form-control py-4" id="inputPassword" type="password"
                                                placeholder="Ingresa tu contraseña" name="password" />
                                        </div>
                                        <div class="form-group">
                                            <div class="custom-control custom-checkbox"><input
                                                    class="custom-control-input" id="rememberPasswordCheck"
                                                    type="checkbox" /><label class="custom-control-label"
                                                    for="rememberPasswordCheck">Recordar contraseña</label></div>
                                        </div>
                                        <div
                                            class="form-group d-flex align-items-center justify-content-between mt-4 mb-0">
                                            <a class="small" href="password.html">Contraseña olvidada?</a>
                                            <button type="submit" class="btn btn-primary">Iniciar Sesion</button>
                                        </div>
                                    </form>
                                </div>

                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>

    </div>
    <script src="https://code.jquery.com/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.bundle.min.js" crossorigin="anonymous">
    </script>
    <script src="{{ asset('template/js/scripts.js') }}"></script>
</body>

</html>
