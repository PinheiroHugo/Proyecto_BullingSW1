<!DOCTYPE html>
<html lang="en">

<head>
    <base href="{{ asset('../../../../') }}">
    <meta charset="utf-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <meta name="description" content="" />
    <meta name="author" content="" />
    <title>@yield('title')</title>
    <link href="{{ asset('template/css/styles.css') }}" rel="stylesheet" />
    <link href="https://cdn.datatables.net/1.10.20/css/dataTables.bootstrap4.min.css" rel="stylesheet"
        crossorigin="anonymous" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.11.2/js/all.min.js" crossorigin="anonymous"></script>
</head>

<body class="sb-nav-fixed">
    <nav class="sb-topnav navbar navbar-expand navbar-dark bg-dark">
        <a class="navbar-brand" href="index.html">EDUCA-SCHOOL</a><button class="btn btn-link btn-sm order-1 order-lg-0"
            id="sidebarToggle" href="#"><i class="fas fa-bars"></i></button><!-- Navbar Search-->
        <a class="d-none d-md-inline-block form-inline ml-auto mr-0 mr-md-3 my-2 my-md-0"></a>
        <!-- Navbar-->
        <ul class="navbar-nav ml-auto ml-md-0">
            <li class="nav-item dropdown">
                <a class="nav-link dropdown-toggle" id="userDropdown" href="#" role="button"
                    data-toggle="dropdown" aria-haspopup="true" aria-expanded="false"><i
                        class="fas fa-user fa-fw"></i></a>
                <div class="dropdown-menu dropdown-menu-right" aria-labelledby="userDropdown">
                    <a class="dropdown-item" href="#">Configuraciones</a><a class="dropdown-item"
                        href="#">Log De
                        Actividades</a>
                    <div class="dropdown-divider"></div>
                    <form action="{{ route('admin.logout') }}" method="POST">
                        {{ csrf_field() }}
                        <button class="dropdown-item">Cerrar Sesion</button>
                    </form>

                </div>
            </li>
        </ul>
    </nav>
    <div id="layoutSidenav">
        <div id="layoutSidenav_nav">
            <nav class="sb-sidenav accordion sb-sidenav-dark" id="sidenavAccordion">
                <div class="sb-sidenav-menu">
                    <div class="nav">
                        <div class="sb-sidenav-menu-heading">Nucleo</div>
                        <a class="nav-link" href="{{ route('admin.dashboard') }}">
                            <div class="sb-nav-link-icon"><i class="fas fa-tachometer-alt"></i></div>
                            Reportes
                        </a>

                        <div class="sb-sidenav-menu-heading">Administracion</div>

                        <a class="nav-link" href="{{ route('admin.informes.index') }}">
                            <div class="sb-nav-link-icon">
                                <i class="fas fa-columns"></i>
                            </div>
                            Ver informes de eventos
                        </a>

                        {{-- <a class="nav-link" href="{{ route('admin.vendedores') }}">
                            <div class="sb-nav-link-icon">
                                <i class="fas fa-book-open"></i>
                            </div>
                            Vendedores
                        </a>

                        <a class="nav-link" href="{{ route('admin.transacciones') }}">
                            <div class="sb-nav-link-icon">
                                <i class="fas fa-dollar-sign"></i>
                            </div>
                            Retiros
                        </a> --}}


                    </div>
                </div>
                <div class="sb-sidenav-footer">
                    <div class="small">Inicio de Sesion como:</div>
                    Administrador
                </div>
            </nav>
        </div>
        <div id="layoutSidenav_content">
            <main>
                @yield('content')
            </main>
            <footer class="py-4 bg-light mt-auto">
                <div class="container-fluid">
                    <div class="d-flex align-items-center justify-content-between small">
                        <div class="text-muted">Copyright &copy; Taller de Grado 2023</div>
                        <div>
                            <a href="#">Politica de privacidad</a>
                            &middot;
                            <a href="#">Terminos &amp; condiciones</a>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.bundle.min.js" crossorigin="anonymous">
    </script>
    <script src="{{ asset('template/js/scripts.js') }}"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.min.js" crossorigin="anonymous"></script>
    <script src="{{ asset('template/assets/demo/chart-area-demo.js') }}"></script>
    <script src="{{ asset('template/assets/demo/chart-bar-demo.js') }}"></script>
    <script src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.datatables.net/1.10.20/js/dataTables.bootstrap4.min.js" crossorigin="anonymous"></script>
    {{-- <script src="{{ asset('template/assets/demo/datatables-demo.js') }}"></script> --}}

    @yield('scripts')
</body>

</html>
