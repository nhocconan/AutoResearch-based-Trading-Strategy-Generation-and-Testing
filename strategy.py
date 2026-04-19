#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1-day ATR filter and volume confirmation.
# Uses Donchian(20) breakout for trend capture, 1-day ATR for volatility filtering,
# and volume surge to confirm breakout strength. Designed for low frequency
# (target ~20-40 trades/year) to minimize fee drag. Works in bull via breakouts
# and in bear via short breakdowns with volatility filter avoiding whipsaws.
# Long when price breaks above Donchian(20) high and ATR(1d) rising with volume spike.
# Short when price breaks below Donchian(20) low and ATR(1d) rising with volume spike.
# Exit on opposite Donchian touch or ATR contraction.
# Strict conditions to limit trades and avoid overtrading.
name = "4h_Donchian20_ATR1d_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) on daily
    atr_1d = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Align ATR to 4h timeframe (waits for completed daily bar)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with rising ATR and volume spike
            if (close[i] > high_max[i] and 
                atr_1d_aligned[i] > atr_1d_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with rising ATR and volume spike
            elif (close[i] < low_min[i] and 
                  atr_1d_aligned[i] > atr_1d_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches Donchian low or ATR falls
            if (close[i] < low_min[i]) or (atr_1d_aligned[i] < atr_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches Donchian high or ATR falls
            if (close[i] > high_max[i]) or (atr_1d_aligned[i] < atr_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals