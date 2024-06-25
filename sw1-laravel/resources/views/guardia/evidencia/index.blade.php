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
                                <th>Archivo</th>
                                <th>Tipo</th>
                            </tr>
                        </thead>
                        <tfoot>
                            <tr>
                                <th>ID</th>
                                <th>Fecha</th>
                                <th>Archivo</th>
                                <th>Tipo</th>
                            </tr>
                        </tfoot>
                        <tbody>

                            @foreach ($evidencias as $p)
                                <tr>
                                    <td>{{ $p->id }}</td>
                                    <td>{{ $p->created_at }}</td>
                                    {{-- @if ($p->tipo == 'Video')
                                        <td>
                                            <video width="320" height="240" controls>
                                                <source src="{{ asset('storage/' . $p->file) }}" type="video/avi">
                                                Your browser does not support the video tag.
                                            </video>
                                        </td>
                                    @else
                                        <td>
                                            <audio controls>
                                                <source src="{{ asset('storage/' . $p->file) }}" type="video/mp4">
                                                Your browser does not support the video tag.
                                            </audio>
                                        </td>
                                    @endif --}}
                                    <td>
                                    <a href="{{ Storage::disk('s3')->url('' . $p->file) }}" target="_blank" download>Descargar evidencia</a>
                                    </td>
                                    <td>{{$p->tipo}}</td>
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
