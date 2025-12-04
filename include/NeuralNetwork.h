#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Sigmoid.h"
#include "ExpectedMovement.h"

// --- CONFIGURAÇÕES ---
#define PadroesTreinamento 36 
#define PadroesValidacao 36 
#define Sucesso 0.01            
#define NumeroCiclos 200000     

#define TaxaAprendizado 0.4     
#define Momentum 0.9            
#define MaximoPesoInicial 0.5

// --- OBJETIVOS (EXTREMOS PARA EVITAR DÚVIDAS) ---
#define OUT_DR_DIREITA    0.10    
#define OUT_DR_FRENTE     0.50    
#define OUT_DR_ESQUERDA   0.90    

// Movimento
#define OUT_DM_FRENTE     0.20      
#define OUT_DM_RE         0.80

// Ângulo
#define OUT_AR_SEM_ROTACAO  0.05
#define OUT_AR_SUAVE        0.30  
#define OUT_AR_FORTE        0.80  

#define ALCANCE_MAX_SENSOR 5000
#define NodosEntrada 8
#define NodosOcultos 12  
#define NodosSaida 3

class NeuralNetwork {
public:
    // --- VARIÁVEIS DE CONTROLE (RESTAURADAS) ---
    int i, j, p, q, r;
    
    // Esta variável controla os prints no terminal (Erro corrigido aqui)
    int IntervaloTreinamentosPrintTela; 
    
    int IndiceRandom[PadroesTreinamento];
    long CiclosDeTreinamento;
    
    // Variável usada na inicialização de pesos
    float Rando; 
    
    float Error;
    float AcumulaPeso; 
    // -------------------------------------------

    // Auxiliares da Rede
    float Oculto[NodosOcultos];
    float PesosCamadaOculta[NodosEntrada + 1][NodosOcultos];
    float OcultoDelta[NodosOcultos];
    float AlteracaoPesosOcultos[NodosEntrada + 1][NodosOcultos];
    ActivationFunction* activationFunctionCamadasOcultas;

    float Saida[NodosSaida];
    float SaidaDelta[NodosSaida];
    float PesosSaida[NodosOcultos + 1][NodosSaida];
    float AlterarPesosSaida[NodosOcultos + 1][NodosSaida];
    ActivationFunction* activationFunctionCamadaSaida;

    float ValoresSensores[1][NodosEntrada] = {{0}};

    // ==================================================================================
    // 36 EXEMPLOS DE TREINAMENTO (BALANCEADOS)
    // ==================================================================================
    const float Input[PadroesTreinamento][NodosEntrada] = {
        // G1: LIVRE/CORREDOR -> FRENTE
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 01
        {3000, 5000, 5000, 5000, 5000, 5000, 5000, 3000}, // 02
        { 800,  800, 5000, 5000, 5000, 5000,  800,  800}, // 03
        { 600,  600, 5000, 5000, 5000, 5000,  600,  600}, // 04
        {1000, 5000, 5000, 5000, 5000, 5000, 5000, 1000}, // 05
        {4000, 4000, 5000, 5000, 5000, 5000, 4000, 4000}, // 06
        { 700,  700, 5000, 5000, 5000, 5000,  900,  900}, // 07
        { 900,  900, 5000, 5000, 5000, 5000,  700,  700}, // 08
        {5000, 5000, 5000, 4000, 4000, 5000, 5000, 5000}, // 09
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 10
        { 800,  800, 5000, 5000, 5000, 5000,  800,  800}, // 11
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 12

        // G2: OBS ESQUERDA -> DIREITA (0.10)
        { 400, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 13
        { 500,  500, 5000, 5000, 5000, 5000, 5000, 5000}, // 14
        {5000, 5000,  400,  400, 5000, 5000, 5000, 5000}, // 15
        { 800,  800,  800, 5000, 5000, 5000, 5000, 5000}, // 16
        { 300, 5000, 5000, 5000, 5000, 5000, 2000, 2000}, // 17
        { 500,  500, 1000, 5000, 5000, 5000, 5000, 5000}, // 18
        { 400,  400,  400,  400, 5000, 5000, 5000, 5000}, // 19
        {5000, 5000, 5000,  400,  400, 5000, 5000, 5000}, // 20
        {5000, 5000, 1000,  500,  500, 1000, 5000, 5000}, // 21
        { 300, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 22
        {5000, 5000,  500,  500, 5000, 5000, 5000, 5000}, // 23
        {1000, 1000, 5000, 5000, 5000, 5000, 5000, 5000}, // 24

        // G3: OBS DIREITA -> ESQUERDA (0.90)
        {5000, 5000, 5000, 5000, 5000, 5000, 5000,  400}, // 25
        {5000, 5000, 5000, 5000, 5000, 5000,  500,  500}, // 26
        {5000, 5000, 5000, 5000,  400,  400, 5000, 5000}, // 27
        {5000, 5000, 5000, 5000, 5000,  800,  800,  800}, // 28
        {2000, 2000, 5000, 5000, 5000, 5000, 5000,  300}, // 29
        {5000, 5000, 5000, 5000, 5000, 1000,  500,  500}, // 30
        {5000, 5000, 5000, 5000,  400,  400,  400,  400}, // 31
        { 400,  400,  400,  400,  400,  400,  400,  400}, // 32
        { 400, 5000, 5000,  400,  400, 5000, 5000,  400}, // 33
        {5000, 5000, 5000, 5000, 5000, 5000, 5000,  300}, // 34
        {5000, 5000, 5000, 5000,  500,  500, 5000, 5000}, // 35
        {5000, 5000, 5000, 5000, 5000, 5000, 1000, 1000}  // 36
    };
    
    float InputNormalizado[PadroesTreinamento][NodosEntrada];

    // ==================================================================================
    // OBJETIVOS 
    // ==================================================================================
    const float Objetivo[PadroesTreinamento][NodosSaida] = {
        // G1: FRENTE
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},

        // G2: DIREITA
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},

        // G3: ESQUERDA
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_RE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}
    };
    
    // Validacao
    const float InputValidacao[PadroesValidacao][NodosEntrada] = {
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, {3000, 5000, 5000, 5000, 5000, 5000, 5000, 3000},
        { 800,  800, 5000, 5000, 5000, 5000,  800,  800}, { 600,  600, 5000, 5000, 5000, 5000,  600,  600},
        {1000, 5000, 5000, 5000, 5000, 5000, 5000, 1000}, {4000, 4000, 5000, 5000, 5000, 5000, 4000, 4000},
        { 700,  700, 5000, 5000, 5000, 5000,  900,  900}, { 900,  900, 5000, 5000, 5000, 5000,  700,  700},
        {5000, 5000, 5000, 4000, 4000, 5000, 5000, 5000}, {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
        { 800,  800, 5000, 5000, 5000, 5000,  800,  800}, {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
        { 400, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, { 500,  500, 5000, 5000, 5000, 5000, 5000, 5000},
        {5000, 5000,  400,  400, 5000, 5000, 5000, 5000}, { 800,  800,  800, 5000, 5000, 5000, 5000, 5000},
        { 300, 5000, 5000, 5000, 5000, 5000, 2000, 2000}, { 500,  500, 1000, 5000, 5000, 5000, 5000, 5000},
        { 400,  400,  400,  400, 5000, 5000, 5000, 5000}, {5000, 5000, 5000,  400,  400, 5000, 5000, 5000},
        {5000, 5000, 1000,  500,  500, 1000, 5000, 5000}, { 300, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
        {5000, 5000,  500,  500, 5000, 5000, 5000, 5000}, {1000, 1000, 5000, 5000, 5000, 5000, 5000, 5000},
        {5000, 5000, 5000, 5000, 5000, 5000, 5000,  400}, {5000, 5000, 5000, 5000, 5000, 5000,  500,  500},
        {5000, 5000, 5000, 5000,  400,  400, 5000, 5000}, {5000, 5000, 5000, 5000, 5000,  800,  800,  800},
        {2000, 2000, 5000, 5000, 5000, 5000, 5000,  300}, {5000, 5000, 5000, 5000, 5000, 1000,  500,  500},
        {5000, 5000, 5000, 5000,  400,  400,  400,  400}, { 400,  400,  400,  400,  400,  400,  400,  400},
        { 400, 5000, 5000,  400,  400, 5000, 5000,  400}, {5000, 5000, 5000, 5000, 5000, 5000, 5000,  300},
        {5000, 5000, 5000, 5000,  500,  500, 5000, 5000}, {5000, 5000, 5000, 5000, 5000, 5000, 1000, 1000}
    };
    
    float InputValidacaoNormalizado[PadroesValidacao][NodosEntrada];
    
    const float ObjetivoValidacao[PadroesValidacao][NodosSaida] = {
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_RE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}
    };

public:
    NeuralNetwork();
    void treinarRedeNeural();
    void inicializacaoPesos();
    int treinoInicialRede();
    void PrintarValores();
    ExpectedMovement testarValor();
    ExpectedMovement definirAcao(int sensor0, int sensor1, int sensor2, int sensor3, int sensor4, int sensor5, int sensor6, int sensor7);
    void validarRedeNeural();
    void treinarValidar();
    void normalizarEntradas();
    void setupCamadas() ;
};

#endif // NEURALNETWORK_H