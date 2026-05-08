#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + daily EMA34 trend + volume spike
# Uses daily EMA34 for trend bias, 12h Donchian breakout for entry signal,
# and 12h volume spike (>2x 20-period average) for confirmation.
# Designed to capture trend continuation with confirmation. Target: 12-37 trades/year.

name = "12h_Donchian20_EMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donch_high[i] = np.max(high[i-20:i])
            donch_low[i] = np.min(low[i-20:i])
    
    # Calculate 12h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA34 to 12h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema34_daily_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for long: price breaks above Donchian high + above daily EMA + volume spike
            if close[i] > donch_high[i] and close[i] > ema34_daily_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Look for short: price breaks below Donchian low + below daily EMA + volume spike
            elif close[i] < donch_low[i] and close[i] < ema34_daily_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low or below daily EMA or volume drops
            if close[i] < donch_low[i] or close[i] < ema34_daily_aligned[i] or volume[i] <= 2.0 * vol_avg_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or above daily EMA or volume drops
            if close[i] > donch_high[i] or close[i] > ema34_daily_aligned[i] or volume[i] <= 2.0 * vol_avg_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals