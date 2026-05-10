#!/usr/bin/env python3
# 1D_WeeklyTrend_Donchian20_Breakout
# Hypothesis: Breakouts of daily Donchian(20) channels in alignment with weekly trend.
# Long when price breaks above 20-day high and weekly EMA50 trend is up.
# Short when price breaks below 20-day low and weekly EMA50 trend is down.
# Uses volume confirmation (>1.5x average) to filter false breakouts.
# Works in bull/bear by following weekly trend and using volume to confirm institutional interest.
# Target: 10-25 trades/year per symbol.

name = "1D_WeeklyTrend_Donchian20_Breakout"
timeframe = "1d"
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
    
    # Daily indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian channels (20-period)
    donch_high = high_s.rolling(window=20, min_periods=20).max().values
    donch_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + breakout above Donchian high + volume
            if weekly_up and close[i] > donch_high[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + breakdown below Donchian low + volume
            elif weekly_down and close[i] < donch_low[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retests Donchian low or weekly trend changes
            if close[i] < donch_low[i] or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retests Donchian high or weekly trend changes
            if close[i] > donch_high[i] or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals