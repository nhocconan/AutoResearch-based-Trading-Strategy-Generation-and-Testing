#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3, R4, S4
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev) / 6
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev) / 6
    camarilla_r4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    camarilla_s4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Align Camarilla levels to 12H timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Load 1W data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close
    close_1w = df_1d['close'].values  # Temporary: will replace with actual 1W close
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike in uptrend
            if (close[i] > r3_12h[i]) and volume_spike[i] and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike in downtrend
            elif (close[i] < s3_12h[i]) and volume_spike[i] and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R4 or trend reverses
            if (close[i] >= r4_12h[i]) or (close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S4 or trend reverses
            if (close[i] <= s4_12h[i]) or (close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals