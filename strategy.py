#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week trend filter and volume confirmation
# The 1D timeframe targets long-term trends while avoiding excessive trading.
# Donchian breakouts capture momentum moves, filtered by weekly trend to avoid counter-trend trades.
# Volume confirms the breakout strength. Designed for 10-30 trades/year to minimize fee drag.

name = "1d_Donchian20_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_avg_today = vol_avg_20[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg_today
        
        if position == 0:
            # Long entry: price breaks above Donchian high + weekly uptrend + volume
            if price > donchian_high[i] and price > ema20_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + weekly downtrend + volume
            elif price < donchian_low[i] and price < ema20_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes
            if price < donchian_low[i] or price < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes
            if price > donchian_high[i] or price > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals