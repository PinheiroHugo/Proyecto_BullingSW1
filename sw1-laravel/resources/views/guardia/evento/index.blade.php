@extends('guardia.layouts.template')

@section('content')
    <div class="container-fluid">
        <h1 class="mt-4">Eventos</h1>
        <ol class="breadcrumb mb-4">
            <li class="breadcrumb-item active">Eventos</li>
        </ol>



        <div class="card mb-4">
            <div class="card-header"><i class="fas fa-table mr-1"></i>Informes</div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Fecha</th>
                                <th>Descripcion</th>
                                <th>Camara</th>
                                <th>Queja de Trabajador</th>
                                <th>Ver evidencias</th>
                                <th>Realizar informe</th>
                            </tr>
                        </thead>
                        <tfoot>
                            <tr>
                                <th>ID</th>
                                <th>Fecha</th>
                                <th>Descripcion</th>
                                <th>Camara</th>
                                <th>Queja de Trabajador</th>
                                <th>Ver evidencias</th>
                                <th>Realizar informe</th>
                            </tr>
                        </tfoot>
                        <tbody>

                            @foreach ($eventos as $p)
                                <tr>
                                    <td>{{ $p->id }}</td>
                                    <td>{{ $p->created_at }}</td>
                                    <td>{{ $p->descripcion }}</td>
                                    <td>{{ $p->camara }}</td>
                                    <td>Sistema</td>
                                    <td><a class="btn btn-primary" href="{{route('guardia.evidencia.index',$p->id)}}">Evidencias</a></td>
                                    <td><a class="btn btn-primary" href="{{route('guardia.informe.crear',$p->id)}}">Realizar Informe</a></td>
                                </tr>
                            @endforeach

                        </tbody>
                    </table>
                </div>
            </div>
        </div>

    </div>
@endsection


@section('scripts')
    <script>
        $(document).ready(function() {
            $('#dataTable').DataTable();
        });
    </script>
@endsection
