<?php

namespace App\Http\Requests\Evento;

use Illuminate\Foundation\Http\FormRequest;

class StoreEventoRequest extends FormRequest
{
    /**
     * Indicates if the validator should stop on the first rule failure.
     *
     * @var bool
     */
    protected $stopOnFirstFailure = true;

    /**
     * Determine if the user is authorized to make this request.
     *
     * @return bool
     */
    public function authorize()
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array
     */
    public function rules()
    {
        return [
            'fecha'=>['bail','required','date'],
            'descripcion'=>['bail','required','min:1', 'max:250'],
            'es_queja'=>['bail','required','max:1'],
            'trabajador_id'=>['bail','exists:trabajadors,id'],
            'camara_id'=>['bail','required','exists:camaras,id'],

        ];
    }

    /**
     * Get custom attributes for validator errors.
     *
     * @return array
     */
    public function attributes()
    {
        return [
            'fecha'=>'',
            'descripcion'=>'',
            'es_queja'=>'',
            'trabajador_id'=>'',
            'camara_id'=>'',

        ];
    }

    /**
     * Get the error messages for the defined validation rules.
     *
     * @return array
     */
    public function messages()
    {
        return [];
    }
}
