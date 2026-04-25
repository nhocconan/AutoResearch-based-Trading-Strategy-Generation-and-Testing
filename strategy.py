#!/usr/bin/env python3
"""
1d Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation
Hypothesis: Daily Donchian breakouts capture intermediate-term trends in BTC/ETH.
1w EMA50 filter ensures we only trade in the direction of the weekly trend,
reducing whipsaw during ranging or countertrend periods.
Volume spike confirmation adds conviction to breakouts.
Designed for 1d timeframe with 7-25 trades/year (30-100 total over 4 years)
to minimize fee drag while capturing strong trends in both bull and bear markets.
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
    
    # Get 1d data for Donchian channels and EMA50 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 for EMA50 + enough for Donchian
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    donchian_high_20 = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_1d.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        close_1w = pd.Series(df_1w['close'])
        ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Align 1d indicators to 1d timeframe (no shift needed as we're already on 1d)
    donchian_high_20_aligned = donchian_high_20  # Already on 1d
    donchian_low_20_aligned = donchian_low_20    # Already on 1d
    ema_50_1d_aligned = ema_50_1d                # Already on 1d
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA50, and volume MA
    start_idx = max(50, 20)  # 50 for EMA50, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_high_20_aligned[i]
        donchian_low = donchian_low_20_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakouts in direction of weekly trend
            weekly_uptrend = curr_close > ema_50_1w_val
            weekly_downtrend = curr_close < ema_50_1w_val
            
            if weekly_uptrend:
                # Uptrend: look for long breakouts above Donchian high
                long_signal = (curr_close > donchian_high) and volume_confirm
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
                    position = 0
            elif weekly_downtrend:
                # Downtrend: look for short breakouts below Donchian low
                short_signal = (curr_close < donchian_low) and volume_confirm
                if short_signal:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
            else:
                # Price near weekly EMA50: stay flat to avoid whipsaw
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below 1d EMA50 OR Donchian low breaks
            if curr_close < ema_50_1d_val or curr_low < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above 1d EMA50 OR Donchian high breaks
            if curr_close > ema_50_1d_val or curr_high > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0