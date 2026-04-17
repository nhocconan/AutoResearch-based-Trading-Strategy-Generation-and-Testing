#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter and volatility context
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, above weekly EMA, and volatility not extreme
            if (close[i] > high_20[i] and volume_filter[i] and 
                close[i] > ema_21_1w_aligned[i] and 
                atr_14_1w_aligned[i] < np.roll(atr_14_1w_aligned, 1)[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, below weekly EMA, and volatility not extreme
            elif (close[i] < low_20[i] and volume_filter[i] and 
                  close[i] < ema_21_1w_aligned[i] and 
                  atr_14_1w_aligned[i] < np.roll(atr_14_1w_aligned, 1)[i] * 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or volatility spikes
            if (close[i] < low_20[i] or 
                atr_14_1w_aligned[i] > np.roll(atr_14_1w_aligned, 1)[i] * 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or volatility spikes
            if (close[i] > high_20[i] or 
                atr_14_1w_aligned[i] > np.roll(atr_14_1w_aligned, 1)[i] * 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_WeeklyEMA21_ATRFilter"
timeframe = "1d"
leverage = 1.0