#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = ema34_1w > np.roll(ema34_1w, 1)
    weekly_trend_up = np.concatenate([[False], weekly_trend_up[1:]])
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    
    # Get 1D data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_base = prev_close
    
    # Resistance levels
    r3 = camarilla_base + range_1d * 1.1 / 6
    r4 = camarilla_base + range_1d * 1.1 / 2
    # Support levels
    s3 = camarilla_base - range_1d * 1.1 / 6
    s4 = camarilla_base - range_1d * 1.1 / 2
    
    # Align pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above R3 with weekly uptrend and volume
        if (close[i] > r3_aligned[i] and 
            weekly_trend_up_aligned[i] and 
            volume[i] > 1.5 * vol_ma20[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below S3 with weekly downtrend and volume
        elif (close[i] < s3_aligned[i] and 
              not weekly_trend_up_aligned[i] and 
              volume[i] > 1.5 * vol_ma20[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses back through R4/S4 or loses weekly trend
        elif position == 1:
            if (close[i] < r4_aligned[i] or not weekly_trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if (close[i] > s4_aligned[i] or weekly_trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals