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
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily ATR 20-period moving average
    atr_series = pd.Series(atr_1d)
    atr_ma_20_1d = atr_series.rolling(window=20, min_periods=20).mean().values
    
    # Align daily ATR and its MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    # Calculate 4-period volume moving average on 4h timeframe
    volume_series = pd.Series(volume)
    vol_ma_4 = volume_series.rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.2 * 20-period ATR MA (high volatility regime)
        vol_filter = atr_1d_aligned[i] > 1.2 * atr_ma_20_1d_aligned[i]
        
        # Volume filter: current 4-period volume MA > 1.5 * its 20-period MA
        vol_ma_4_20 = np.full(n, np.nan)
        if i >= 23:  # 4 + 19 for 20-period MA of vol_ma_4
            vol_ma_4_20[i] = np.mean(vol_ma_4[i-19:i+1])
        vol_filter_vol = (not np.isnan(vol_ma_4_20[i]) and 
                         vol_ma_4[i] > 1.5 * vol_ma_4_20[i])
        
        # Entry conditions: volatility expansion + volume surge
        long_entry = vol_filter and vol_filter_vol
        short_entry = vol_filter and vol_filter_vol
        
        # Exit conditions: volatility contraction or volume drop
        vol_exit = (atr_1d_aligned[i] < 0.8 * atr_ma_20_1d_aligned[i]) or \
                   (not np.isnan(vol_ma_4_20[i]) and vol_ma_4[i] < 0.8 * vol_ma_4_20[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and vol_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and vol_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_volatility_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0