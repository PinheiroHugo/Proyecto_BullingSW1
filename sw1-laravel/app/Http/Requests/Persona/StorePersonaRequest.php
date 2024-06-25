<?php

namespace App\Http\Requests\Persona;

use Illuminate\Foundation\Http\FormRequest;

class StorePersonaRequest extends FormRequest
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
            'nombre'=>['bail','required','min:1', 'max:250'],
            'apellido_p'=>['bail','required','min:1', 'max:250'],
            'apellido_m'=>['bail','required','min:1', 'max:250'],
            'ci'=>['bail','required','min:1', 'max:250'],
            'telefono'=>['bail','required','min:1', 'max:250'],
            'foto'=>['bail','file', 'max:10240','mimes:jpg,bmp,png,jpeg'],
            'user_id'=>['bail','required','exists:users,id'],

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
            'nombre'=>'',
            'apellido_p'=>'',
            'apellido_m'=>'',
            'ci'=>'',
            'telefono'=>'',
            'foto'=>'',
            'user_id'=>'',

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
