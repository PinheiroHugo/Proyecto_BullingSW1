<?php

namespace App\Http\Requests\Informe;

use Illuminate\Foundation\Http\FormRequest;

class StoreInformeRequest extends FormRequest
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
            'titulo'=>['bail','required','min:1', 'max:250'],
            'documento'=>['bail','file', 'max:10240'],
            'guardia_id'=>['bail','required','exists:guardias,id'],
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
            'titulo'=>'',
            'documento'=>'',
            'guardia_id'=>'',
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
