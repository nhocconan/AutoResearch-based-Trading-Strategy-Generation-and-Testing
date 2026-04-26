#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dRegime_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d volatility regime (ATR ratio) and volume spike. Uses tighter Camarilla levels (R1/S1) for selective entries. ATR regime filter distinguishes high volatility (breakout favorable) from low volatility (false breakouts likely) markets. Volume spike confirms institutional participation. Designed for 15-30 trades/year to minimize fee drag while working in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volatility regime: ATR ratio (current ATR / 20-period average ATR)
    atr_ma = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d_aligned / atr_ma  # >1 = high vol, <1 = low vol
    high_vol_regime = atr_ratio > 1.2  # High volatility regime favorable for breakouts
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 12.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 20, 14)  # volume avg, ATR, ATR ratio
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(high_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout with volume confirmation AND high volatility regime
            # Long: break above R1 + volume spike + high vol regime
            long_entry = (close_val > R1_aligned[i]) and volume_spike[i] and high_vol_regime[i]
            # Short: break below S1 + volume spike + high vol regime
            short_entry = (close_val < S1_aligned[i]) and volume_spike[i] and high_vol_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement or ATR stoploss (2.0 * ATR)
            exit_condition = (close_val < S1_aligned[i]) or \
                           (close_val < entry_price - 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or ATR stoploss (2.0 * ATR)
            exit_condition = (close_val > R1_aligned[i]) or \
                           (close_val > entry_price + 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dRegime_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0