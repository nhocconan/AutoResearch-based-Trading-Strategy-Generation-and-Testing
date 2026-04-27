#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Spike and ADX Trend Filter.
Long when: 1) Price breaks above 20-week Donchian upper, 2) ADX(14) > 25 (trending), 3) Volume > 1.5x 20-day average.
Short when: 1) Price breaks below 20-week Donchian lower, 2) ADX(14) > 25 (trending), 3) Volume > 1.5x 20-day average.
Exit when price returns to weekly midpoint (mean reversion) or ADX < 20 (range).
Designed for 1d timeframe: targets 30-100 total trades over 4 years (7-25/year).
Works in both bull (breakouts) and bear (breakdowns) with trend filter to avoid whipsaws.
"""

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
    
    # Get weekly data for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 20-week Donchian channels
    lookback = 20
    upper_20w = np.full(len(high_1w), np.nan)
    lower_20w = np.full(len(low_1w), np.nan)
    for i in range(lookback, len(high_1w)):
        upper_20w[i] = np.max(high_1w[i-lookback+1:i+1])
        lower_20w[i] = np.min(low_1w[i-lookback+1:i+1])
    
    upper_20w_aligned = align_htf_to_ltf(prices, df_1w, upper_20w)
    lower_20w_aligned = align_htf_to_ltf(prices, df_1w, lower_20w)
    
    # Weekly midpoint for exit
    midpoint_20w = (upper_20w + lower_20w) / 2.0
    midpoint_20w_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20w)
    
    # ADX(14) on weekly for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] > 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] > 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) > 0 else 0
        
        # ADX is smoothed DX
        adx = np.zeros(len(high))
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_14w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14w_aligned = align_htf_to_ltf(prices, df_1w, adx_14w)
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Donchian (20), ADX (28), volume MA (20)
    start_idx = max(28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20w_aligned[i]) or np.isnan(lower_20w_aligned[i]) or 
            np.isnan(midpoint_20w_aligned[i]) or np.isnan(adx_14w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        upper = upper_20w_aligned[i]
        lower = lower_20w_aligned[i]
        midpoint = midpoint_20w_aligned[i]
        adx = adx_14w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: ADX > 25 for trending market
        trend_filter = adx > 25
        
        # Exit trend filter: ADX < 20 for ranging market
        exit_trend_filter = adx < 20
        
        if position == 0:
            # Long: price breaks above upper Donchian + trend + volume
            if price > upper and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + trend + volume
            elif price < lower and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint OR ADX < 20 (range)
            if price <= midpoint or exit_trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint OR ADX < 20 (range)
            if price >= midpoint or exit_trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0