#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Sigmoid.h"
#include "ExpectedMovement.h"

// --- CONFIGURAÇÕES DA REDE ---
#define PadroesTreinamento 44 
#define PadroesValidacao 44 

#define Sucesso 0.01            
#define NumeroCiclos 200000     
 
#define TaxaAprendizado 0.2     
#define Momentum 0.9            
#define MaximoPesoInicial 0.5

// --- OBJETIVOS (OUTPUTS) ---
#define OUT_DR_DIREITA    0.10    
#define OUT_DR_FRENTE     0.50    
#define OUT_DR_ESQUERDA   0.90    

// Movimento
#define OUT_DM_FRENTE     0.20      
#define OUT_DM_RE         0.80

// Ângulo de Rotação
#define OUT_AR_SEM_ROTACAO  0.05
#define OUT_AR_SUAVE        0.30  
#define OUT_AR_FORTE        0.80  

#define ALCANCE_MAX_SENSOR 5000
#define NodosEntrada 8
#define NodosOcultos 14  
#define NodosSaida 3

class NeuralNetwork {
public:
    // --- VARIÁVEIS DE CONTROLE ---
    int i, j, p, q, r;
    int IntervaloTreinamentosPrintTela; 
    int IndiceRandom[PadroesTreinamento];
    long CiclosDeTreinamento;
    float Rando; 
    float Error;
    float AcumulaPeso; 

    // --- ESTRUTURA DA REDE ---
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
 
    // 44 EXEMPLOS DE TREINAMENTO
    const float Input[PadroesTreinamento][NodosEntrada] = { 
        
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 01 - Espaço Aberto Perfeito
        {3000, 5000, 5000, 5000, 5000, 5000, 5000, 3000}, // 02 - Corredor Largo
        { 800,  800, 5000, 5000, 5000, 5000,  800,  800}, // 03 - Corredor Estreito (Alinhado)
        { 600,  600, 5000, 5000, 5000, 5000,  600,  600}, // 04 - Corredor Muito Estreito
        {1000, 5000, 5000, 5000, 5000, 5000, 5000, 1000}, // 05 - Entrando em corredor
        {4000, 4000, 5000, 5000, 5000, 5000, 4000, 4000}, // 06 - Sala ampla
        { 700,  700, 5000, 5000, 5000, 5000,  900,  900}, // 07 - Leve desalinho (mas frente livre)
        { 900,  900, 5000, 5000, 5000, 5000,  700,  700}, // 08 - Leve desalinho (mas frente livre)
          
        // Ação: FRENTE. O robô aprende a focar nos sensores centrais para decisão de avançar.
        {2000, 2000, 4000, 5000, 5000, 4000, 2000, 2000}, // 09 - Quarto afunilando
        {1500, 3000, 5000, 5000, 5000, 5000, 3000, 1500}, // 10 - Passando por uma porta/fresta
        {5000, 2500, 5000, 5000, 5000, 5000, 2500, 5000}, // 11 - Obstáculos laterais médios
        {3000, 3000, 3000, 5000, 5000, 3000, 3000, 3000}, // 12 - Sala pequena, saída à frente

        // --- G2: OBSTÁCULO ESQUERDA -> VIRAR DIREITA---
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

        // --- G3: OBSTÁCULO DIREITA -> VIRAR ESQUERDA---
        {5000, 5000, 5000, 5000, 5000, 5000, 5000,  400}, // 25
        {5000, 5000, 5000, 5000, 5000, 5000,  500,  500}, // 26
        {5000, 5000, 5000, 5000,  400,  400, 5000, 5000}, // 27
        {5000, 5000, 5000, 5000, 5000,  800,  800,  800}, // 28
        {2000, 2000, 5000, 5000, 5000, 5000, 5000,  300}, // 29
        {5000, 5000, 5000, 5000, 5000, 1000,  500,  500}, // 30
        {5000, 5000, 5000, 5000,  400,  400,  400,  400}, // 31
        
        // EMERGÊNCIA: Tudo fechado -> Ré Reto
        { 400,  400,  400,  400,  400,  400,  400,  400}, // 32
        
        { 400, 5000, 5000,  400,  400, 5000, 5000,  400}, // 33
        {5000, 5000, 5000, 5000, 5000, 5000, 5000,  300}, // 34
        {5000, 5000, 5000, 5000,  500,  500, 5000, 5000}, // 35
        {5000, 5000, 5000, 5000, 5000, 5000, 1000, 1000}, // 36

        // --- G4: CORREÇÕES SUAVES DIREITA---
        // Aqui o robô aprende: "Se um lado está meio perto, mas a frente está livre, corrija suave"
        {2500, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 37
        {2500, 2500, 5000, 5000, 5000, 5000, 5000, 5000}, // 38
        {2200, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 39
        {3000, 3000, 5000, 5000, 5000, 5000, 5000, 5000}, // 40

        // --- G5: CORREÇÕES SUAVES ESQUERDA---
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 2500}, // 41
        {5000, 5000, 5000, 5000, 5000, 5000, 2500, 2500}, // 42
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 2200}, // 43
        {5000, 5000, 5000, 5000, 5000, 5000, 3000, 3000}  // 44
    };
    
    float InputNormalizado[PadroesTreinamento][NodosEntrada];
 
    const float Objetivo[PadroesTreinamento][NodosSaida] = {
        // G1: FRENTE (1-12) - Agora inclui saida de quartos
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // Pattern 09 (Funil)
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // Pattern 10 (Porta)
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // Pattern 11 (Passagem)
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, // Pattern 12 (Sala)

        // G2: DIREITA FORTE (13-24)
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

        // G3: ESQUERDA FORTE (25-36)
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_RE},     // 32 RE RETO
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},

        // G4: DIREITA SUAVE (37-40)
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},

        // G5: ESQUERDA SUAVE (41-44)
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}
    };
      
    const float InputValidacao[PadroesValidacao][NodosEntrada] = {
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, {3000, 5000, 5000, 5000, 5000, 5000, 5000, 3000},
        { 800,  800, 5000, 5000, 5000, 5000,  800,  800}, { 600,  600, 5000, 5000, 5000, 5000,  600,  600},
        {1000, 5000, 5000, 5000, 5000, 5000, 5000, 1000}, {4000, 4000, 5000, 5000, 5000, 5000, 4000, 4000},
        { 700,  700, 5000, 5000, 5000, 5000,  900,  900}, { 900,  900, 5000, 5000, 5000, 5000,  700,  700},
        
        // (saída de quarto)
        {2000, 2000, 4000, 5000, 5000, 4000, 2000, 2000}, 
        {1500, 3000, 5000, 5000, 5000, 5000, 3000, 1500},
        {5000, 2500, 5000, 5000, 5000, 5000, 2500, 5000}, 
        {3000, 3000, 3000, 5000, 5000, 3000, 3000, 3000},
 
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
        {5000, 5000, 5000, 5000,  500,  500, 5000, 5000}, {5000, 5000, 5000, 5000, 5000, 5000, 1000, 1000},
        {2500, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, {2500, 2500, 5000, 5000, 5000, 5000, 5000, 5000},
        {2200, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, {3000, 3000, 5000, 5000, 5000, 5000, 5000, 5000},
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 2500}, {5000, 5000, 5000, 5000, 5000, 5000, 2500, 2500},
        {5000, 5000, 5000, 5000, 5000, 5000, 5000, 2200}, {5000, 5000, 5000, 5000, 5000, 5000, 3000, 3000}
    };
    
    float InputValidacaoNormalizado[PadroesValidacao][NodosEntrada];
    
    const float ObjetivoValidacao[PadroesValidacao][NodosSaida] = {
        // Mantido exatamente igual ao Objetivo[] para garantir validação correta
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE, OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
        {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE}, {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_FRENTE},
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
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_FRENTE,   OUT_AR_SEM_ROTACAO, OUT_DM_RE}, // 32
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_FORTE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_DIREITA,  OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE},
        {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}, {OUT_DR_ESQUERDA, OUT_AR_SUAVE,       OUT_DM_FRENTE}
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