#!/usr/bin/env python3
name = "6h_WeeklyPivot_DonchianBreakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Weekly pivot (from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Use previous week's high/low/close for pivot calculation
    high_prev_1w = df_1w['high'].values
    low_prev_1w = df_1w['low'].values
    close_prev_1w = df_1w['close'].values
    pivot = (high_prev_1w + low_prev_1w + close_prev_1w) / 3.0
    r1 = 2 * pivot - low_prev_1w
    s1 = 2 * pivot - high_prev_1w
    r2 = pivot + (high_prev_1w - low_prev_1w)
    s2 = pivot - (high_prev_1w - low_prev_1w)
    
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day (4*6h) to reduce trade frequency
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above Donchian high with volume in 1d uptrend AND above weekly R1
            if (close[i] > donchian_high[i] and 
                trending_up and 
                vol_confirm[i] and 
                close[i] > r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below Donchian low with volume in 1d downtrend AND below weekly S1
            elif (close[i] < donchian_low[i] and 
                  trending_down and 
                  vol_confirm[i] and 
                  close[i] < s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below Donchian low or 1d trend changes to down
            if close[i] < donchian_low[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or 1d trend changes to up
            if close[i] > donchian_high[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot provides institutional reference points, Donchian breakout captures momentum,
# volume confirms institutional participation, and 1d trend filter ensures alignment with higher timeframe.
# In bull markets: 1d trend up, breakouts above Donchian high with weekly R1 support capture continuation.
# In bear markets: 1d trend down, breakdowns below Donchian low with weekly S1 resistance capture continuation.
# Weekly pivot adds structural context beyond daily levels, reducing false breakouts.
# 6h timeframe balances responsiveness with lower trade frequency (~20-40 trades/year target).