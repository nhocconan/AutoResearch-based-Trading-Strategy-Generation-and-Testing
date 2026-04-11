#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Donchian breakout + volume confirmation + ADX trend filter.
# Uses 1-day Donchian channels for breakout entries, volume filter to confirm institutional participation,
# and ADX to filter for trending conditions. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns.
# ADX filter prevents whipsaws in ranging markets.

name = "12h_1d_donchian_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band = highest high over past 20 days
    upper_20 = np.full(len(high_1d), np.nan)
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
    
    # Lower band = lowest low over past 20 days
    lower_20 = np.full(len(low_1d), np.nan)
    for i in range(19, len(low_1d)):
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate daily ADX (14-period) for trend strength
    # First calculate +DM and -DM
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - high_1d[i-1]),
            abs(low_1d[i] - low_1d[i-1])
        )
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(len(high_1d))
    plus_dm_smooth = np.zeros(len(high_1d))
    minus_dm_smooth = np.zeros(len(high_1d))
    
    # Initial values (first 14 periods)
    if len(high_1d) >= 14:
        atr[13] = np.mean(tr[1:15])
        plus_dm_smooth[13] = np.mean(plus_dm[1:15])
        minus_dm_smooth[13] = np.mean(minus_dm[1:15])
        
        # Wilder smoothing for rest
        for i in range(14, len(high_1d)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(len(high_1d))
    minus_di = np.zeros(len(high_1d))
    dx = np.zeros(len(high_1d))
    
    for i in range(14, len(high_1d)):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(len(high_1d))
    if len(high_1d) >= 27:  # Need 14 for DX + 14 for smoothing
        adx[26] = np.mean(dx[14:28])
        for i in range(27, len(high_1d)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align daily indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = high[i] >= upper_20_aligned[i] and vol_filter and trend_filter
        breakout_short = low[i] <= lower_20_aligned[i] and vol_filter and trend_filter
        
        # Exit conditions: reverse signal or loss of trend
        exit_long = position == 1 and (breakout_short or adx_aligned[i] < 20)
        exit_short = position == -1 and (breakout_long or adx_aligned[i] < 20)
        
        # Update position
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals