#!/usr/bin/env python3
"""
1d_WeeklyTrend_DailyBreakout_v2
Hypothesis: Daily timeframe with weekly trend filter using EMA20 and daily breakout of price channels.
Long when price breaks above daily Donchian upper channel (20) in weekly uptrend with volume confirmation.
Short when price breaks below daily Donchian lower channel (20) in weekly downtrend with volume confirmation.
Uses volume filter (daily volume > 1.5x 20-day average) to avoid false breakouts.
Designed for 10-25 trades/year to minimize fee drag on daily timeframe.
Works in bull/bear via trend filter and breakout logic.
"""

name = "1d_WeeklyTrend_DailyBreakout_v2"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily (no shift needed as already daily)
    donchian_high_aligned = high_max_20  # Already aligned to daily
    donchian_low_aligned = low_min_20    # Already aligned to daily
    
    # Get daily volume for confirmation
    vol_ma20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.divide(volume, vol_ma20_1d, out=np.zeros_like(volume), where=vol_ma20_1d!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        if np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high in weekly uptrend with volume
            if (high[i] > donchian_high_aligned[i] and 
                close[i] > donchian_high_aligned[i] and  # confirmation close
                vol_ratio_1d[i] > 1.5 and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in weekly downtrend with volume
            elif (low[i] < donchian_low_aligned[i] and 
                  close[i] < donchian_low_aligned[i] and  # confirmation close
                  vol_ratio_1d[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low or trend turns down
            if (close[i] < donchian_low_aligned[i]) or \
               not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high or trend turns up
            if (close[i] > donchian_high_aligned[i]) or \
               not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals