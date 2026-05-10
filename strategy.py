# 6h_WeeklyPivot_DailyTrend_Volume_Breakout
# Hypothesis: Weekly pivots (from weekly candles) provide strong support/resistance levels.
# In trending markets (1d EMA50 filter), price breaking above weekly R1 in uptrend or
# below weekly S1 in downtrend continues with momentum. Volume confirmation avoids false breakouts.
# Weekly pivots are more significant than daily pivots, reducing whipsaw in sideways markets.
# Target: 15-35 trades/year to minimize fee drag (6h timeframe).

name = "6h_WeeklyPivot_DailyTrend_Volume_Breakout"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivots from previous week
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly range = weekly high - weekly low
    weekly_range = df_1w['high'] - df_1w['low']
    # Weekly R1 = pivot + (range * 1.1)
    weekly_r1 = weekly_pivot + (weekly_range * 1.1)
    # Weekly S1 = pivot - (range * 1.1)
    weekly_s1 = weekly_pivot - (weekly_range * 1.1)
    
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50), weekly pivots (needs 1w), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R1 + volume
            if uptrend and close[i] > weekly_r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S1 + volume
            elif downtrend and close[i] < weekly_s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters weekly pivot area
            if not uptrend or close[i] < weekly_pivot.iloc[-1] if hasattr(weekly_pivot, 'iloc') else weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters weekly pivot area
            if not downtrend or close[i] > weekly_pivot.iloc[-1] if hasattr(weekly_pivot, 'iloc') else weekly_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals