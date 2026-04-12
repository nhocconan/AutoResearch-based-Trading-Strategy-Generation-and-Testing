#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily data for price levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 14-day ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Align ATR to daily timeframe (since we're using 1d timeframe)
    atr_1d_aligned = atr_1d  # Already at daily frequency
    
    # Calculate weekly trend using SMA crossover (more stable than Ichimoku)
    close_1w = df_1w['close'].values
    sma_20 = np.full(len(close_1w), np.nan)
    sma_50 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20[i] = np.mean(close_1w[i-20:i])
    for i in range(50, len(close_1w)):
        sma_50[i] = np.mean(close_1w[i-50:i])
    weekly_trend = np.where(sma_20 > sma_50, 1, np.where(sma_20 < sma_50, -1, 0))
    
    # Align weekly trend to daily timeframe
    weekly_trend_1d = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate daily pivot points using previous day's data
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: R1, S1, R2, S2
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + range_val
    s2 = pivot - range_val
    
    # Align levels to daily timeframe
    r1_1d = r1
    s1_1d = s1
    r2_1d = r2
    s2_1d = s2
    
    # Calculate volume moving average (20-period on daily)
    vol_ma_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        vol_ma_1d[i] = np.mean(volume[i-20:i])  # volume is already daily aligned
    
    # Align volume MA to price array (since we're using 1d timeframe)
    vol_ma = vol_ma_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to ensure we have enough data
        # Map daily index to price array index (since we're using 1d timeframe)
        # For 1d timeframe, we need to find the corresponding daily bar
        # We'll use the fact that each day has multiple intraday bars
        # But since we're using 1d as primary timeframe, we process once per day
        
        # Skip if data not ready
        if (np.isnan(weekly_trend_1d[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Only take trades in direction of weekly trend
        weekly_bullish = weekly_trend_1d[i] == 1
        weekly_bearish = weekly_trend_1d[i] == -1
        
        # Entry conditions: S1/R1 breakout with volume confirmation and weekly trend alignment
        long_breakout = (close[i] > r1_1d[i]) and volume_filter and weekly_bullish
        short_breakout = (close[i] < s1_1d[i]) and volume_filter and weekly_bearish
        
        # Exit conditions: touch opposite S1/R1 level or weekly trend reversal or ATR stop
        long_exit = (close[i] < s1_1d[i]) or (weekly_trend_1d[i] == -1) or (close[i] < r1_1d[i] - 1.5 * atr_1d[i])
        short_exit = (close[i] > r1_1d[i]) or (weekly_trend_1d[i] == 1) or (close[i] > s1_1d[i] + 1.5 * atr_1d[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_pivot_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0