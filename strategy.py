#!/usr/bin/env python3
"""
12h_1w_camarilla_breakout_volume_v1
Strategy: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Camarilla pivot levels from daily data to identify potential reversal zones, combined with 1-week trend filter and volume confirmation to avoid false breakouts. Designed to capture institutional order flow around key pivot levels in both trending and ranging markets. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # 1d Camarilla pivot levels (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels (using previous day's data)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)
    r2 = pivot + (range_1d * 1.1 / 4)
    r1 = pivot + (range_1d * 1.1 / 6)
    s1 = pivot - (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Shift levels forward by 1 to use previous day's levels for current day
    pivot = np.concatenate([[np.nan], pivot[:-1]])
    r3 = np.concatenate([[np.nan], r3[:-1]])
    r2 = np.concatenate([[np.nan], r2[:-1]])
    r1 = np.concatenate([[np.nan], r1[:-1]])
    s1 = np.concatenate([[np.nan], s1[:-1]])
    s2 = np.concatenate([[np.nan], s2[:-1]])
    s3 = np.concatenate([[np.nan], s3[:-1]])
    
    # Align Camarilla levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = (price_close > r3_12h[i]) and volume_spike[i]
        breakdown_short = (price_close < s3_12h[i]) and volume_spike[i]
        
        # Pullback entries (more frequent)
        pullback_long = (price_close <= r1_12h[i] and price_close >= pivot_12h[i]) and uptrend_1w and volume_spike[i]
        pullback_short = (price_close >= s1_12h[i] and price_close <= pivot_12h[i]) and downtrend_1w and volume_spike[i]
        
        # Exit conditions
        exit_long = position == 1 and (price_close < pivot_12h[i] or not volume_spike[i])
        exit_short = position == -1 and (price_close > pivot_12h[i] or not volume_spike[i])
        
        # Trading logic
        if (breakout_long or pullback_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (breakdown_short or pullback_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses Camarilla pivot levels from daily data to identify potential reversal zones, combined with 1-week trend filter and volume confirmation to avoid false breakouts. Designed to capture institutional order flow around key pivot levels in both trending and ranging markets. Target: 50-150 total trades over 4 years.