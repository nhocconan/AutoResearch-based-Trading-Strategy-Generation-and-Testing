#/usr/bin/env python3
# 12h_1d_Camarilla_Breakout_Volume_v2
# Hypothesis: Breakouts above/below daily Camarilla L4/H4 levels with volume confirmation
# capture momentum from key daily support/resistance. Works in both bull and bear markets
# by filtering with 1w EMA trend (only trade in trend direction) to avoid false breakouts
# in ranging markets. Target: 12-30 trades per year (48-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # === CAMARILLA PIVOT LEVELS (based on previous 1d bar) ===
    # Calculate from previous 1d bar's OHLC
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    
    # First bar will have invalid data, but we'll handle with valid check
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    l3 = pivot + (range_val * 1.1 / 4)
    l4 = pivot + (range_val * 1.1 / 2)
    h3 = pivot - (range_val * 1.1 / 4)
    h4 = pivot - (range_val * 1.1 / 2)
    
    # Align to 12h timeframe (these levels are valid for the entire 1d bar)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === WEEKLY EMA TREND FILTER (50-period) ===
    weekly_ema = np.full(n, np.nan)
    if n >= 50:
        k = 2 / (50 + 1)
        weekly_ema[49] = weekly_close[:50].mean()
        for i in range(50, n):
            weekly_ema[i] = weekly_close[i] * k + weekly_ema[i-1] * (1 - k)
    
    # Weekly EMA aligned to 12h
    weekly_ema_12h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === VOLUME SPIKE (2x 20-period average on 12h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(l3_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(h3_12h[i]) or np.isnan(h4_12h[i]) or
            np.isnan(weekly_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price breaks through Camarilla levels with volume confirmation
        break_long = high[i] > l4_12h[i] and vol_spike[i]  # Break above L4
        break_short = low[i] < h4_12h[i] and vol_spike[i]  # Break below H4
        
        # Trend filter: only take longs above weekly EMA, shorts below
        trend_long = close[i] > weekly_ema_12h[i]
        trend_short = close[i] < weekly_ema_12h[i]
        
        # Exit when price returns to pivot level
        # Signal logic
        if break_long and trend_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif break_short and trend_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and low[i] <= pivot_12h[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] >= pivot_12h[i]:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals