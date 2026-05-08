#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses weekly pivot levels (calculated from weekly OHLC) to establish long-term bias,
# Donchian breakout for entry timing, and volume spike for confirmation.
# Works in bull markets by buying breakouts above weekly pivot resistance,
# and in bear markets by selling breakdowns below weekly pivot support.
# Target: 15-35 trades/year (60-140 total over 4 years).

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, etc.
    # We'll use R4/S4 for breakout signals
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    r4 = r3 + (high_weekly - low_weekly)
    s4 = s3 - (high_weekly - low_weekly)
    
    # Get daily data for volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Align daily volume average to 6h timeframe
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 periods for volume average
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-day average volume
        vol_6h_current = volume[i]
        vol_confirm = vol_6h_current > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly pivot direction and volume confirmation
            
            # Calculate Donchian channels for current bar (lookback 20 periods)
            lookback_start = max(0, i - 19)
            donchian_high = np.max(high[lookback_start:i+1])
            donchian_low = np.min(low[lookback_start:i+1])
            
            # Long when price breaks above Donchian high AND above weekly R4 (bullish bias)
            long_condition = (
                close[i] > donchian_high and      # Donchian breakout
                close[i] > r4_aligned[i] and      # Above weekly R4 (strong bullish bias)
                vol_confirm                       # Volume confirmation
            )
            
            # Short when price breaks below Donchian low AND below weekly S4 (bearish bias)
            short_condition = (
                close[i] < donchian_low and       # Donchian breakdown
                close[i] < s4_aligned[i] and      # Below weekly S4 (strong bearish bias)
                vol_confirm                       # Volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low or weekly pivot
            lookback_start = max(0, i - 19)
            donchian_low = np.min(low[lookback_start:i+1])
            
            if close[i] < donchian_low or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or weekly pivot
            lookback_start = max(0, i - 19)
            donchian_high = np.max(high[lookback_start:i+1])
            
            if close[i] > donchian_high or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals