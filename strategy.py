#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Triple_Confluence_Breakout"
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
    
    # Get 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data (approximate weekly from daily)
    # We'll use 5-day rolling window to simulate weekly from daily data
    high_5d = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Standard pivot: P = (H + L + C)/3
    pivot_5d = (high_5d + low_5d + close_5d) / 3.0
    # Resistance/Support levels
    r1_5d = 2 * pivot_5d - low_5d
    s1_5d = 2 * pivot_5d - high_5d
    
    # Align 5d-derived weekly levels to 4h timeframe
    pivot_5d_aligned = align_htf_to_ltf(prices, df_1d, pivot_5d)
    r1_5d_aligned = align_htf_to_ltf(prices, df_1d, r1_5d)
    s1_5d_aligned = align_htf_to_ltf(prices, df_1d, s1_5d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_5d_aligned[i]) or 
            np.isnan(r1_5d_aligned[i]) or
            np.isnan(s1_5d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_5d_aligned[i]
        r1_val = r1_5d_aligned[i]
        s1_val = s1_5d_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + price above weekly EMA20 + volume filter
            if close[i] > r1_val and close[i] > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S1 + price below weekly EMA20 + volume filter
            elif close[i] < s1_val and close[i] < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below pivot or below weekly EMA20
            if close[i] < pivot_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above pivot or above weekly EMA20
            if close[i] > pivot_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals