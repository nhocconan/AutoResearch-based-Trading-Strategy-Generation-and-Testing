#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian20_WeeklyTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly SuperTrend for Trend Direction ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # SuperTrend calculation
    hl2 = (high_1w + low_1w) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)
    
    # Initialize SuperTrend
    supertrend = np.zeros_like(close_1w)
    uptrend = np.ones_like(close_1w, dtype=bool)
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upperband[i-1]:
            uptrend[i] = True
        elif close_1w[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = lowerband[i] if uptrend[i] else upperband[i]
    
    # Align SuperTrend to 12h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend.astype(float))
    
    # === Daily Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        supertrend_val = supertrend_aligned[i]
        uptrend_val = uptrend_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(supertrend_val) or np.isnan(uptrend_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with weekly uptrend and volume
            if close_val > upper_val and uptrend_val >= 0.5 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with weekly downtrend and volume
            elif close_val < lower_val and uptrend_val < 0.5 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian OR trend turns down
            if close_val < lower_val or uptrend_val < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian OR trend turns up
            if close_val > upper_val or uptrend_val >= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals