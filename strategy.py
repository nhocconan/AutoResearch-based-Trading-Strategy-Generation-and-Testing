#!/usr/bin/env python3
name = "12h_1w_1d_Camarilla_Pivot_Breakout"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # Based on previous day's OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (d_high + d_low + d_close) / 3
    range_val = d_high - d_low
    
    # Camarilla levels: S1, S2, S3, R1, R2, R3
    # S1 = close - (range * 1.1/12)
    # S2 = close - (range * 1.1/6)
    # S3 = close - (range * 1.1/4)
    # R1 = close + (range * 1.1/12)
    # R2 = close + (range * 1.1/6)
    # R3 = close + (range * 1.1/4)
    s1 = d_close - (range_val * 1.1 / 12)
    s2 = d_close - (range_val * 1.1 / 6)
    s3 = d_close - (range_val * 1.1 / 4)
    r1 = d_close + (range_val * 1.1 / 12)
    r2 = d_close + (range_val * 1.1 / 6)
    r3 = d_close + (range_val * 1.1 / 4)
    
    # Weekly trend filter: price above/below weekly EMA34
    weekly_close = df_1w['close'].values
    ema34_w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = weekly_close > ema34_w
    weekly_trend_down = weekly_close < ema34_w
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align all indicators to 12h timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long signals: price breaks above R1, R2, or R3 with weekly uptrend and volume
            if (weekly_trend_up_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > r2_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short signals: price breaks below S1, S2, or S3 with weekly downtrend and volume
            elif (weekly_trend_down_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                elif close[i] < s2_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                elif close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or weekly trend turns down
            if (close[i] < s1_aligned[i] or not weekly_trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or weekly trend turns up
            if (close[i] > r1_aligned[i] or weekly_trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals