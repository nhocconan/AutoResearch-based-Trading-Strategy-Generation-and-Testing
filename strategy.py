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
    
    # Get daily data for price levels and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
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
    
    # Align ATR to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate weekly trend using SMA crossover (more stable than Ichimoku)
    close_1w = df_1w['close'].values
    sma_20 = np.full(len(close_1w), np.nan)
    sma_50 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20[i] = np.mean(close_1w[i-20:i])
    for i in range(50, len(close_1w)):
        sma_50[i] = np.mean(close_1w[i-50:i])
    weekly_trend = np.where(sma_20 > sma_50, 1, np.where(sma_20 < sma_50, -1, 0))
    weekly_trend_12h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
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
    
    # Align levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate volume moving average (20-period on 12h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_12h[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Only take trades in direction of weekly trend
        weekly_bullish = weekly_trend_12h[i] == 1
        weekly_bearish = weekly_trend_12h[i] == -1
        
        # Entry conditions: S1/R1 breakout with volume confirmation and weekly trend alignment
        long_breakout = (close[i] > r1_12h[i]) and volume_filter and weekly_bullish
        short_breakout = (close[i] < s1_12h[i]) and volume_filter and weekly_bearish
        
        # Exit conditions: touch opposite S1/R1 level or weekly trend reversal or ATR stop
        long_exit = (close[i] < s1_12h[i]) or (weekly_trend_12h[i] == -1) or (close[i] < r1_12h[i] - 1.5 * atr_12h[i])
        short_exit = (close[i] > r1_12h[i]) or (weekly_trend_12h[i] == 1) or (close[i] > s1_12h[i] + 1.5 * atr_12h[i])
        
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

name = "12h_1w_1d_pivot_breakout_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0