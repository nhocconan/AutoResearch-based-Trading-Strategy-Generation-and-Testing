#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_keltner_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Keltner Channel (20 EMA, 10 ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA(20) of close
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # Align Keltner Channel to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 6h volume
        vol_current = volume[i]
        vol_surge = vol_current > 1.5 * vol_avg_aligned[i]  # 50% above average
        
        # Long breakout: price closes above upper Keltner with volume surge
        long_signal = (close[i] > upper_aligned[i] and vol_surge)
        # Short breakout: price closes below lower Keltner with volume surge
        short_signal = (close[i] < lower_aligned[i] and vol_surge)
        
        # Exit when price returns to middle (EMA)
        ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
        exit_long = close[i] < ema_20_aligned[i]
        exit_short = close[i] > ema_20_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals