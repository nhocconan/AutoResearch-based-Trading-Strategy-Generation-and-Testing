# Hypothesis: Combine 4h price action with 1d support/resistance levels and 1w trend filter
# Uses 1d pivot points (R1/S1) for entry/exit and 1w SMA crossover for trend filter
# Volume confirmation reduces false breakouts
# Designed for fewer trades (<100/year) to minimize fee drag in both bull and bear markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
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
    
    # Align ATR to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate weekly trend using SMA crossover (20/50)
    close_1w = df_1w['close'].values
    sma_20 = np.full(len(close_1w), np.nan)
    sma_50 = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_20[i] = np.mean(close_1w[i-20:i])
    for i in range(50, len(close_1w)):
        sma_50[i] = np.mean(close_1w[i-50:i])
    weekly_trend = np.where(sma_20 > sma_50, 1, np.where(sma_20 < sma_50, -1, 0))
    weekly_trend_4h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate daily pivot points using previous day's data
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: R1, S1
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume moving average (20-period on 4h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_4h[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Only take trades in direction of weekly trend
        weekly_bullish = weekly_trend_4h[i] == 1
        weekly_bearish = weekly_trend_4h[i] == -1
        
        # Entry conditions: S1/R1 breakout with volume confirmation and weekly trend alignment
        long_breakout = (close[i] > r1_4h[i]) and volume_filter and weekly_bullish
        short_breakout = (close[i] < s1_4h[i]) and volume_filter and weekly_bearish
        
        # Exit conditions: touch opposite S1/R1 level or weekly trend reversal or ATR stop
        long_exit = (close[i] < s1_4h[i]) or (weekly_trend_4h[i] == -1) or (close[i] < r1_4h[i] - 2.0 * atr_4h[i])
        short_exit = (close[i] > r1_4h[i]) or (weekly_trend_4h[i] == 1) or (close[i] > s1_4h[i] + 2.0 * atr_4h[i])
        
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

name = "4h_1w_1d_pivot_breakout_weekly_trend_v2"
timeframe = "4h"
leverage = 1.0