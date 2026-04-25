#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Weekly EMA50 Trend + Volume Spike
Hypothesis: Daily Donchian breakout captures intermediate-term moves, weekly EMA50 filters trend direction, volume spike confirms participation. Works in bull (long when price breaks above Donchian upper + price > weekly EMA50 + volume spike) and bear (short when price breaks below Donchian lower + price < weekly EMA50 + volume spike). Targets 30-100 trades over 4 years on 1d timeframe.
"""

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
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1d timeframe (identity for same timeframe, but keep for consistency)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Get 1w data for weekly EMA50 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian (20) and volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema_50 = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper, price > weekly EMA50, volume confirmation
            long_entry = curr_close > upper and curr_close > ema_50 and volume_confirm
            # Short: price breaks below Donchian lower, price < weekly EMA50, volume confirmation
            short_entry = curr_close < lower and curr_close < ema_50 and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower OR price falls below weekly EMA50
            if curr_close < lower or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper OR price rises above weekly EMA50
            if curr_close > upper or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0