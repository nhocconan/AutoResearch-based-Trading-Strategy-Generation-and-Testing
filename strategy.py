#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: Daily Donchian(20) breakouts capture strong trends. Weekly EMA50 filter ensures alignment with major trend.
Volume spike (>2.0x 20-day volume MA) confirms breakout strength. Designed for 1d timeframe to minimize fee drag
while working in both bull (breakouts with trend) and bear (breakdowns with trend) markets.
Target: 15-25 trades/year to stay within fee limits.
"""

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
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50 and volume MA
    start_idx = max(51, 20)  # 51 for EMA50 (50 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Donchian channels using 1d data (lookback 20 periods)
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakouts in direction of trend
            if price_above_ema:
                # Uptrend: look for long breakouts above Donchian high
                long_signal = (curr_high > donchian_high) and volume_confirm
            elif price_below_ema:
                # Downtrend: look for short breakdowns below Donchian low
                short_signal = (curr_low < donchian_low) and volume_confirm
            else:
                # Price exactly at EMA (rare) - no signal
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or reverses below EMA
            if curr_low < donchian_low or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or reverses above EMA
            if curr_high > donchian_high or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0