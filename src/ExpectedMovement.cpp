#include "ExpectedMovement.h"
#include "Config.h"
#include <math.h>
#include <iostream>

ExpectedMovement::ExpectedMovement(float _direcaoRotacao, float _direcaoMovimento, float _anguloRotacao)
{
    this->DirecaoRotacao = _direcaoRotacao;
    this->DirecaoMovimento = _direcaoMovimento;
    this->AnguloRotacao = _anguloRotacao;
}

void ExpectedMovement::ProcessarMovimento()
{
    // 1. VELOCIDADE (Mantém rápido)
    float velocidadeReal = VELOCIDADEDESLOCAMENTO * 1.5; 

    // 2. FORÇA DO GIRO
    float anguloBase = 0.0;
    if (this->AnguloRotacao < 0.20)         
        anguloBase = 0.0; 
    else if (this->AnguloRotacao < 0.55)    
        anguloBase = 15.0; 
    else                                    
        anguloBase = 30.0;

    // 3. DECISÃO DE DIREÇÃO (AJUSTE DE SENSIBILIDADE)
    // Alvo da Rede: Dir=0.10 | Frente=0.50 | Esq=0.90
    
    // DIREITA - Aumentei a margem para pegar qualquer sinal abaixo de 0.40
    if (this->DirecaoRotacao < 0.40) 
    {
        this->DirecaoRotacaoProcessada = 2; 
        this->AnguloRotacaoProcessado = -anguloBase; 
        
        // Se a rede pediu Direita mas esqueceu do angulo, força um pouquinho
        if(anguloBase == 0.0) this->AnguloRotacaoProcessado = -15.0;
    }
    // ESQUERDA - Diminui a margem para pegar qualquer sinal acima de 0.60
    // Antes estava 0.70, por isso ele parava de virar.
    else if (this->DirecaoRotacao > 0.60) 
    {
        this->DirecaoRotacaoProcessada = 1; 
        this->AnguloRotacaoProcessado = anguloBase; 
        
        // Se a rede pediu Esquerda mas esqueceu do angulo, força um pouquinho
        if(anguloBase == 0.0) this->AnguloRotacaoProcessado = 15.0;
    }
    // FRENTE - Só segue reto se o valor for EXATAMENTE o meio (entre 0.40 e 0.60)
    else 
    {
        this->DirecaoRotacaoProcessada = 0; 
        this->AnguloRotacaoProcessado = 0.0;
    }

    // 4. MOVIMENTO
    if (this->DirecaoMovimento > 0.60) 
        this->DirecaoMovimentoProcessada = -velocidadeReal;
    else 
        this->DirecaoMovimentoProcessada = velocidadeReal;
        
    // DEBUG NO TERMINAL (Isso vai te ajudar a ver o valor exato)
    // Descomente a linha abaixo se quiser ver os numeros correndo na tela
    // printf("Rotacao RAW: %.2f  |  Decisao: %d\n", this->DirecaoRotacao, (int)this->DirecaoRotacaoProcessada);
}