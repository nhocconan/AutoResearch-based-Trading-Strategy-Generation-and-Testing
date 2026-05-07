#!/usr/bin/env python3
name = "6h_WeeklyPivot_Momentum_1dTrend_Volume"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Weekly pivot levels (using previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_prev_1w = df_1w['high'].values
    low_prev_1w = df_1w['low'].values
    close_prev_1w = df_1w['close'].values
    range_1w = high_prev_1w - low_prev_1w
    
    # Weekly pivot points
    pivot = (high_prev_1w + low_prev_1w + close_prev_1w) / 3
    r1 = 2 * pivot - low_prev_1w
    s1 = 2 * pivot - high_prev_1w
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_surge = volume > (1.5 * vol_ma_24)
    
    # 60-minute momentum (4-period on 6m timeframe)
    mom_4 = np.full(n, np.nan)
    for i in range(4, n):
        mom_4[i] = close[i] - close[i-4]
    
    signals = np.zeros(n)
    position = 0
    bars_since_last_trade = 0
    cooldown_bars = 4  # 24 hours cooldown
    
    start_idx = max(24, 4)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_24[i]) or 
            np.isnan(mom_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price above weekly pivot, bullish momentum, 1d uptrend, volume surge
            if (close[i] > pivot_aligned[i] and 
                mom_4[i] > 0 and 
                trend_up[i] and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price below weekly pivot, bearish momentum, 1d downtrend, volume surge
            elif (close[i] < pivot_aligned[i] and 
                  mom_4[i] < 0 and 
                  trend_down[i] and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price below weekly pivot or momentum turns bearish
            if close[i] < pivot_aligned[i] or mom_4[i] <= 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above weekly pivot or momentum turns bullish
            if close[i] > pivot_aligned[i] or mom_4[i] >= 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot acts as key support/resistance. In 1d uptrend, longs from pivot with momentum and volume capture continuation. In 1d downtrend, shorts from pivot work similarly. Weekly pivot provides structure, 6h momentum captures short-term strength, volume surge confirms institutional interest. Designed for ~50-120 trades over 4 years.