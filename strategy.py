#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1w EMA34 trend filter and volume confirmation.
Long when Alligator is bullish (Lips > Teeth > Jaw) AND 1w EMA34 rising AND volume > 1.3x 20-period average.
Short when Alligator is bearish (Lips < Teeth < Jaw) AND 1w EMA34 falling AND volume > 1.3x 20-period average.
Exit when Alligator convergence (|Lips-Jaw| < 0.1*ATR) or opposite crossover.
Uses 1w HTF for EMA34 trend to avoid whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator (12h timeframe)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    sma_jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    sma_teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    sma_lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # ATR for convergence exit
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 
                    34, 20, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_jaw[i]) or np.isnan(sma_teeth[i]) or np.isnan(sma_lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw = sma_jaw[i]
        teeth = sma_teeth[i]
        lips = sma_lips[i]
        ema_val = ema_34_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_1w_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Alligator conditions
        alligator_bullish = lips > teeth and teeth > jaw
        alligator_bearish = lips < teeth and teeth < jaw
        alligator_convergence = abs(lips - jaw) < 0.1 * atr_val
        
        if position == 0:
            # Long: Bullish Alligator AND EMA34 rising AND volume spike
            if alligator_bullish and ema_rising and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND EMA34 falling AND volume spike
            elif alligator_bearish and ema_falling and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bearish crossover OR Alligator convergence
                if not alligator_bullish or alligator_convergence:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bullish crossover OR Alligator convergence
                if not alligator_bearish or alligator_convergence:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0