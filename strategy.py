#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot S1/R1 breakout with volume confirmation and 1-day trend filter.
Trades only during high-volume breakouts in the direction of the daily trend.
Designed to work in both bull and bear markets by using the daily trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
"""
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4-hour data for Camarilla pivots and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Camarilla levels (based on previous period's range)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels use previous period's close and range
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = close_4h[0]  # first value
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    
    range_4h = prev_high - prev_low
    # Camarilla levels: S1 = C - (H-L)*1.1/12, R1 = C + (H-L)*1.1/12
    s1 = prev_close - range_4h * 1.1 / 12
    r1 = prev_close + range_4h * 1.1 / 12
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla levels, volume MA, and 1d EMA
    start_idx = max(20, 34)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        s1_level = s1_aligned[i]
        r1_level = r1_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average (moderate to balance trades)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Camarilla breakout with volume and daily trend alignment
        if position == 0:
            # Long: break above R1 + volume + daily uptrend
            if close[i] > r1_level and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below S1 + volume + daily downtrend
            elif close[i] < s1_level and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below daily EMA or S1 level
            if close[i] < trend_1d or close[i] < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above daily EMA or R1 level
            if close[i] > trend_1d or close[i] > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_CamarillaS1R1_Volume_1dTrendFilter"
timeframe = "4h"
leverage = 1.0