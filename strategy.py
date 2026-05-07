#!/usr/bin/env python3
"""
1d_WeeklyTrend_DailyBreakout
Hypothesis: Buy weekly uptrend breakouts above daily Donchian(20) high with volume confirmation,
and sell weekly downtrend breakouts below daily Donchian(20) low. Uses weekly trend filter
to avoid counter-trend trades, designed for 15-25 trades/year to minimize fee drag.
Works in bull/bear via weekly trend filter.
"""

name = "1d_WeeklyTrend_DailyBreakout"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # Calculate 20-period volume average for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i]) or
            vol_ma20[i] == 0):
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
            
        trend_up = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above Donchian high + volume
            if trend_up and close[i] > donchian_high[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price breaks below Donchian low + volume
            elif trend_down and close[i] < donchian_low[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly trend turns down OR price breaks below Donchian low
            if not trend_down or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend turns up OR price breaks above Donchian high
            if not trend_up or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals