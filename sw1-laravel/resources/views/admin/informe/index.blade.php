@extends('admin.layouts.template')

@section('content')
    <div class="container-fluid">
        <h1 class="mt-4">Informes</h1>
        <ol class="breadcrumb mb-4">
            <li class="breadcrumb-item active">Informes</li>
        </ol>



        <div class="card mb-4">
            <div class="card-header"><i class="fas fa-table mr-1"></i>Informes</div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-bordered" id="dataTable" width="100%" cellspacing="0">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Titulo</th>
                                <th>Documento</th>
                                <th>Estado</th>
                            </tr>
                        </thead>
                        <tfoot>
                            <tr>
                                <th>ID</th>
                                <th>Titulo</th>
                                <th>Documento</th>
                                <th>Estado</th>
                            </tr>
                        </tfoot>
                        <tbody>

                            @foreach ($informes as $p)
                                <tr>
                                    <td>{{ $p->id }}</td>
                                    <td>{{ $p->titulo }}</td>
                                    <!-- <td><a href="{{ asset('storage/' . $p->documento) }}">Ver Documento</a></td> -->
                                    <td><a href="{{ Storage::disk('s3')->url('' . $p->documento) }}" target="_blank"
                                            download>Descargar informe</a></td>
                                    <td>Estado</td>
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
