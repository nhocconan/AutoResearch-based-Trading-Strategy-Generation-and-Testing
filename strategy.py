#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Uses daily EMA34 for trend bias, Donchian channels for breakout signals,
# and volume surge for entry confirmation. Designed to capture trends in both bull and bear markets
# while avoiding false breakouts in choppy conditions. Target: 12-37 trades/year on 12h timeframe.

name = "12h_Donchian20_1dEMA34_VolumeConfirmation"
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
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate daily volume average for volume confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 12h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
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
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for 12h timeframe (20-period)
        donchian_high = np.max(high[i-19:i+1])
        donchian_low = np.min(low[i-19:i+1])
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average of daily volume
        vol_confirm = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_confirm = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout in direction of daily EMA trend with volume confirmation
            long_condition = (
                close[i] > donchian_high and    # price breaks above Donchian high
                close[i] > ema34_daily_aligned[i] and   # price above EMA34 (bullish bias)
                vol_confirm                     # volume confirmation
            )
            
            short_condition = (
                close[i] < donchian_low and     # price breaks below Donchian low
                close[i] < ema34_daily_aligned[i] and   # price below EMA34 (bearish bias)
                vol_confirm                     # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian mid-point or trend reverses
            donchian_mid = (donchian_high + donchian_low) / 2
            if close[i] < donchian_mid or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian mid-point or trend reverses
            donchian_mid = (donchian_high + donchian_low) / 2
            if close[i] > donchian_mid or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals