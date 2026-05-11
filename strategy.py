#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d_vals + range_1d * 1.1 / 2
    camarilla_s3 = close_1d_vals - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0x 24-period average (24*12h = 12 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 2.0 * vol_ma24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R3 + 1d uptrend + volume confirmation
            if close[i] > camarilla_r3_aligned[i-1] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 + 1d downtrend + volume confirmation
            elif close[i] < camarilla_s3_aligned[i-1] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below S3 OR 1d trend turns down
            if close[i] < camarilla_s3_aligned[i-1] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above R3 OR 1d trend turns up
            if close[i] > camarilla_r3_aligned[i-1] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals