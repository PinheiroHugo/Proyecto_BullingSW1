@extends('guardia.layouts.template')

@section('content')
    <div class="container-fluid">
        <h1 class="mt-4">Subir Informe</h1>
        <ol class="breadcrumb mb-4">
            <li class="breadcrumb-item active">Subir Informe</li>
        </ol>




        @if ($errors->any())
            @foreach ($errors->all() as $error)
                <div class="alert alert-danger">{{ $error }}</div>
            @endforeach
        @endif

        <form action="{{ route('guardia.informe.store',$evento->id) }}" method="POST" enctype="multipart/form-data">
            {{ csrf_field() }}
            
            <input type="text" name="evento_id" value="{{$evento->id}}" hidden>

            <div class="mb-3">
                <label for="exampleFormControlInput9" class="form-label">Titulo</label>
                <input type="text" class="form-control" id="exampleFormControlInput9" name="titulo"
                    placeholder="Introduzca el titulo del informe">
            </div>
            
            <div class="mb-3">
                <label for="exampleFormControlInput9" class="form-label">Documento</label>
                <input type="file" class="form-control" id="exampleFormControlInput9" name="documento">
            </div>

            <br>

            <button class="btn btn-primary">Guardar Informe</button>

        </form>

    </div>
@endsection
