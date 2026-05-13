#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    def calculate_camarilla(high, low, close):
        # Typical price for the day
        typical = (high + low + close) / 3
        # Range
        rng = high - low
        # Camarilla levels
        # R4 = close + rng * 1.1/2
        # R3 = close + rng * 1.1/4
        # S3 = close - rng * 1.1/4
        # S4 = close - rng * 1.1/2
        r3 = close + rng * 1.1 / 4
        s3 = close - rng * 1.1 / 4
        return r3, s3
    
    r3_1d, s3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12H timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Load 1W data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 20-period EMA on weekly
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: 20-period average volume on 1D
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_ok = volume[i] > vol_avg_1d[i] * 1.5
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume and weekly uptrend
            if close[i] > r3_12h[i] and vol_ok and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume and weekly downtrend
            elif close[i] < s3_12h[i] and vol_ok and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below S3 or weekly trend turns down
            if close[i] < s3_12h[i] or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above R3 or weekly trend turns up
            if close[i] > r3_12h[i] or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals