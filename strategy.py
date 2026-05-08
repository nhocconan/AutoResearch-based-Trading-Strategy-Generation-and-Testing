#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# Uses daily EMA50 for trend bias, Donchian(20) breakout for entry, and volume spike (>1.5x average) for confirmation.
# Designed to capture trends in both bull and bear markets while avoiding false breakouts in low-volume conditions.
# Target: 25-40 trades/year (~100-160 total over 4 years).

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
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
    
    # Calculate daily volume average for volume confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
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
        if (np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period) for breakout
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.nan
            lowest_low = np.nan
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average of daily volume
        vol_confirm = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_confirm = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout in direction of daily trend with volume confirmation
            # Long when price breaks above Donchian high in bullish trend
            long_condition = (
                close[i] > highest_high and           # price breaks above Donchian high
                close[i] > ema50_daily_aligned[i] and # price above daily EMA50 (bullish bias)
                vol_confirm                           # volume confirmation
            )
            
            # Short when price breaks below Donchian low in bearish trend
            short_condition = (
                close[i] < lowest_low and             # price breaks below Donchian low
                close[i] < ema50_daily_aligned[i] and # price below daily EMA50 (bearish bias)
                vol_confirm                           # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint or trend reverses
            midpoint = (highest_high + lowest_low) / 2
            if close[i] < midpoint or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint or trend reverses
            midpoint = (highest_high + lowest_low) / 2
            if close[i] > midpoint or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals