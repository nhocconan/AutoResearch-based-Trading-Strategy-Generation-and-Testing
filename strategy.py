#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_period = 14
    tr_series = pd.Series(tr)
    atr14 = tr_series.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr14 = pd.Series(atr14).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    # High-Low range over 14 periods
    max_high = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    range_maxmin = max_high - min_low
    
    # Choppiness Index
    chop = np.zeros(len(close_1d))
    chop[:] = np.nan
    valid = (sum_atr14 > 0) & (~np.isnan(sum_atr14)) & (~np.isnan(range_maxmin))
    chop[valid] = 100 * np.log10(sum_atr14[valid] / range_maxmin[valid]) / np.log10(atr_period)
    
    # Align Choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Chop > 61.8 (ranging) + price breaks above recent high + up-trend on 1w
            if chop_aligned[i] > 61.8 and close[i] > np.max(high[i-20:i]) and vol_ok and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Chop > 61.8 (ranging) + price breaks below recent low + down-trend on 1w
            elif chop_aligned[i] > 61.8 and close[i] < np.min(low[i-20:i]) and vol_ok and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Chop < 38.2 (trending) OR trend reversal OR stop via opposite signal
            if chop_aligned[i] < 38.2 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Chop < 38.2 (trending) OR trend reversal OR stop via opposite signal
            if chop_aligned[i] < 38.2 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals