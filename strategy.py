#!/usr/bin/env python3
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
    
    # Get 1w and 1d data for weekly pivot and 1d EMA
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    
    # Weekly resistance/support levels (similar to Camarilla but simpler)
    r1 = weekly_pivot + (weekly_range * 1.0)
    s1 = weekly_pivot - (weekly_range * 1.0)
    r2 = weekly_pivot + (weekly_range * 1.5)
    s2 = weekly_pivot - (weekly_range * 1.5)
    
    # Align weekly levels to 6h
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot, EMA, volume MA
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume (reduces trade frequency)
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_6h[i]
        bearish_trend = price < ema_50_6h[i]
        
        r1_level = r1_6h[i]
        s1_level = s1_6h[i]
        r2_level = r2_6h[i]
        s2_level = s2_6h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume + bullish trend
            if price > r1_level and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume + bearish trend
            elif price < s1_level and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 or trend turns bearish
            if price < s1_level or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 or trend turns bullish
            if price > r1_level or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0