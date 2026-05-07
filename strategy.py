#!/usr/bin/env python3
name = "1d_TripleBandBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly trend: EMA10 slope
    ema10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_slope = ema10_1w - np.roll(ema10_1w, 1)
    ema10_slope[0] = 0
    ema10_slope_aligned = align_htf_to_ltf(prices, df_1w, ema10_slope)
    
    # Daily Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    ma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + 2 * std20
    lower_bb = ma20 - 2 * std20
    
    # Daily Donchian (20)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume filter: > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema10_slope_aligned[i]) or np.isnan(ma20[i]) or np.isnan(std20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper BB OR upper Donchian with weekly uptrend and volume
            if ((close[i] > upper_bb[i] or close[i] > high_roll[i]) and 
                ema10_slope_aligned[i] > 0 and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB OR lower Donchian with weekly downtrend and volume
            elif ((close[i] < lower_bb[i] or close[i] < low_roll[i]) and 
                  ema10_slope_aligned[i] < 0 and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below lower BB or weekly trend turns down
            if close[i] < lower_bb[i] or ema10_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above upper BB or weekly trend turns up
            if close[i] > upper_bb[i] or ema10_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals