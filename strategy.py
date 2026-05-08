#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d EMA Trend + Volume Spike
# Uses daily EMA50 for trend bias, Donchian(20) breakout for entry, and volume > 1.5x 20-period average for confirmation.
# Designed to capture trend continuations in both bull and bear markets while avoiding false breakouts in low volume.
# Target: 25-50 trades/year to stay within fee-efficient range.

name = "4h_Donchian_EMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate daily volume average for volume spike filter
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Pre-calculate Donchian channels (20-period) for 4h timeframe
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-20:i])
            lowest_low_20[i] = np.min(low[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 20-period average of daily volume
        vol_spike = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume average
            vol_spike = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of daily EMA trend with volume spike
            # Long when price breaks above Donchian upper band in bullish trend
            long_condition = (
                close[i] > highest_high_20[i] and   # price breaks above 20-period high
                close[i] > ema50_daily_aligned[i] and   # price above daily EMA50 (bullish bias)
                vol_spike                             # volume spike for confirmation
            )
            
            # Short when price breaks below Donchian lower band in bearish trend
            short_condition = (
                close[i] < lowest_low_20[i] and     # price breaks below 20-period low
                close[i] < ema50_daily_aligned[i] and   # price below daily EMA50 (bearish bias)
                vol_spike                             # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend reverses
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < donchian_middle or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend reverses
            donchian_middle = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > donchian_middle or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals