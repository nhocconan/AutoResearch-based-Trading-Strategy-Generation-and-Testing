#!/usr/bin/env python3
name = "1d_WilliamsAlligator_ElderRay_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Data for trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1w EMA13 for trend (fast) ===
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # === 13-period EMA for Elder Ray calculation ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === Williams Alligator components (13,8,5 SMAs) ===
    smma13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Jaw
    smma8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values      # Teeth
    smma5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values      # Lips
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 13, 20, 8, 5)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(smma13[i]) or
            np.isnan(smma8[i]) or
            np.isnan(smma5[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Williams Alligator: Mouth open when Teeth > Jaw (bullish) or Teeth < Jaw (bearish)
        # Lips > Teeth > Jaw = strong uptrend, Lips < Teeth < Jaw = strong downtrend
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + price above weekly EMA + Alligator bullish alignment + volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema13_1w_aligned[i] and
                smma5[i] > smma8[i] and  # Lips above Teeth
                smma8[i] > smma13[i] and  # Teeth above Jaw
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + price below weekly EMA + Alligator bearish alignment + volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema13_1w_aligned[i] and
                  smma5[i] < smma8[i] and  # Lips below Teeth
                  smma8[i] < smma13[i] and  # Teeth below Jaw
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power < 0 (selling pressure takes over) or price below weekly EMA or Alligator turns bearish
            if (bear_power[i] < 0 or 
                close[i] < ema13_1w_aligned[i] or
                smma5[i] < smma8[i] or  # Lips below Teeth
                smma8[i] < smma13[i]):  # Teeth below Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (buying pressure takes over) or price above weekly EMA or Alligator turns bullish
            if (bull_power[i] > 0 or 
                close[i] > ema13_1w_aligned[i] or
                smma5[i] > smma8[i] or  # Lips above Teeth
                smma8[i] > smma13[i]):  # Teeth above Jaw
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals