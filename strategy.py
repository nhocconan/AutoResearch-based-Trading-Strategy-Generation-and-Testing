#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day ATR volatility filter and volume confirmation
# Donchian breakouts capture institutional breakout momentum with statistical edge
# ATR filter ensures volatility expansion precedes breakout (avoids low-volatility false breakouts)
# Volume confirmation validates institutional participation
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
# Works in bull/bear via volatility filter and breakout directionality
name = "4h_Donchian20_1dATR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian channel (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: require ATR > 0.5 * 20-period ATR average
        atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_14_1d_aligned[i] > (atr_ma[i] * 0.5) if not np.isnan(atr_ma[i]) else False
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and volatility expansion
            if (close[i] > high_20[i] and 
                volume_confirm[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and volatility expansion
            elif (close[i] < low_20[i] and 
                  volume_confirm[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or ATR collapses
            if (close[i] < low_20[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or ATR collapses
            if (close[i] > high_20[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals