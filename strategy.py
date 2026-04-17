#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for pivots and volume (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Calculate daily ATR(14) for volume filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.append(close_1d[0], close_1d[:-1]))
    low_close = np.abs(low_1d - np.append(close_1d[0], close_1d[:-1]))
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly EMA20 to 12h
    ema20_12h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Align daily pivot levels to 12h
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Align daily ATR14 to 12h
    atr14_12h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: current volume > 2.0 * ATR(14) (proxy for volume spike)
    volume_ma14 = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for EMA20 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema20_12h[i]) or np.isnan(atr14_12h[i]) or np.isnan(volume_ma14[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: volume spike > 2.0 * average volume
        volume_filter = volume[i] > (2.0 * volume_ma14[i])
        
        # Trend filter: price relative to weekly EMA20
        price_above_ema = close[i] > ema20_12h[i]
        price_below_ema = close[i] < ema20_12h[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and above weekly EMA20
            if (close[i] > r1_12h[i] and volume_filter and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and below weekly EMA20
            elif (close[i] < s1_12h[i] and volume_filter and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below R1 or weekly EMA20
            if close[i] < r1_12h[i] or close[i] < ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above S1 or weekly EMA20
            if close[i] > s1_12h[i] or close[i] > ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA20_DailyPivot_Volume"
timeframe = "12h"
leverage = 1.0