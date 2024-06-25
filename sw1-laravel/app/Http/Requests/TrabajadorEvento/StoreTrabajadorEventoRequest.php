<?php

namespace App\Http\Requests\TrabajadorEvento;

use Illuminate\Foundation\Http\FormRequest;

class StoreTrabajadorEventoRequest extends FormRequest
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
            'trabajador_id'=>['bail','required','exists:trabajadors,id'],
            'evento_id'=>['bail','required','exists:eventos,id'],

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
            'trabajador_id'=>'',
            'evento_id'=>'',

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
