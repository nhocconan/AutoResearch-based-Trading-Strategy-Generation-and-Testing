#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ATR regime filter and volume confirmation.
# Uses 6h Donchian(20) for breakout entries, aligned to current bar.
# Long when price breaks above 20-period high with ATR(14) > 1d ATR(14) * 1.2 (high vol regime) and volume > 1.5x 20-period MA.
# Short when price breaks below 20-period low with same filters.
# Volatility regime filter ensures we only trade during elevated volatility, reducing false breakouts in low-vol chop.
# Works in bull/bear via volatility expansion confirmation rather than trend direction.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_1dATR_Volume_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need enough for ATR calculation
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # Align with df_1d index
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h ATR(14) for comparison
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(atr_14_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        atr_1d_val = atr_14_1d_aligned[i]
        atr_6h_val = atr_14_6h[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_spike = volume_spike[i]
        
        # Volatility regime: 6h ATR > 1.2 * 1d ATR (elevated volatility)
        vol_regime = atr_6h_val > (1.2 * atr_1d_val)
        
        # Entry logic
        if position == 0:
            # Long: break above upper channel with volume spike and vol regime
            if close_val > upper_channel and vol_spike and vol_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below lower channel with volume spike and vol regime
            elif close_val < lower_channel and vol_spike and vol_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below lower channel OR vol regime ends
            if close_val < lower_channel or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper channel OR vol regime ends
            if close_val > upper_channel or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals