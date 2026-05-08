#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 55-period Donchian breakout with 1-week trend filter and volume confirmation
# Long when price breaks above upper Donchian channel with uptrend (close > 1w EMA50) and volume spike
# Short when price breaks below lower Donchian channel with downtrend (close < 1w EMA50) and volume spike
# Designed to capture strong directional moves in both bull and bear markets with low trade frequency
# Target: 30-100 total trades over 4 years = 7-25/year

name = "1d_Donchian55_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def donchian_channels(high, low, period):
    """Calculate Donchian channels (upper, lower)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 55-period Donchian channels on 1d data
    upper_dc, lower_dc = donchian_channels(high, low, 55)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_val = upper_dc[i]
        lower_val = lower_dc[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: break above upper Donchian + uptrend + volume spike
            if (close[i] > upper_val and 
                close[i] > ema50_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian + downtrend + volume spike
            elif (close[i] < lower_val and 
                  close[i] < ema50_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR trend reverses
            if close[i] < lower_dc[i] or close[i] < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR trend reverses
            if close[i] > upper_dc[i] or close[i] > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals