#!/usr/bin/env python3
"""
1d_1W_Camarilla_Pivot_Breakout_WeeklyTrend_v1
Hypothesis: On daily chart, buy when price breaks above weekly Camarilla R3 level with above-average volume,
sell when price breaks below weekly S3 level with above-average volume. Exit when price crosses weekly pivot.
Uses weekly trend filter: only take longs when price > weekly EMA20, only shorts when price < weekly EMA20.
Designed for low trade frequency (~10-20/year) to avoid fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Previous week's values for current week's calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    vol_1w = df_1w['volume'].values
    
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Weekly VWAP approximation
    typical_price = (high_1w + low_1w + close_1w) / 3
    vwap_numerator = np.cumsum(typical_price * vol_1w)
    vwap_denominator = np.cumsum(vol_1w)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Weekly Camarilla levels
    range_1w = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pp + (range_1w * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (range_1w * 1.1 / 4)
    
    # Weekly EMA20 for trend filter
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to daily
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
    
    # Volume condition: weekly volume > 20-period average
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current weekly volume > 20-period average
        vol_condition = vol_1w_aligned[i] > vol_ma_20_aligned[i]
        
        # Breakout conditions with weekly trend filter
        long_breakout = close[i] > camarilla_r3_aligned[i] and close[i] > ema_20_aligned[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and close[i] < ema_20_aligned[i]
        
        # Exit condition: price crosses weekly pivot
        long_exit = close[i] < camarilla_pp_aligned[i]
        short_exit = close[i] > camarilla_pp_aligned[i]
        
        if position == 0:
            if long_breakout and vol_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1W_Camarilla_Pivot_Breakout_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0