#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Uses daily EMA34 for trend bias, 12h Donchian channel breakout for entry,
# and volume > 1.5x 20-period average for confirmation. Designed to capture
# trending moves in both bull and bear markets while avoiding false breakouts
# in low-volume conditions. Target: 15-30 trades/year.

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
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
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average of daily volume
        vol_confirm = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_12h_current = volume[i]
            vol_confirm = vol_12h_current > 1.5 * vol_avg_20_daily_aligned[i]
        
        # Calculate 12h Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-19:i+1])
            lowest_low = np.min(low[i-19:i+1])
            donchian_upper = highest_high
            donchian_lower = lowest_low
            
            # Donchian breakout conditions
            long_breakout = close[i] > donchian_upper
            short_breakout = close[i] < donchian_lower
        else:
            long_breakout = False
            short_breakout = False
        
        if position == 0:
            # Look for entry: follow daily EMA trend with Donchian breakout and volume confirmation
            # Long when price above daily EMA34 (bullish bias) and break above upper band
            long_condition = (
                close[i] > ema34_daily_aligned[i] and   # price above EMA34 (bullish bias)
                long_breakout and                       # Donchian breakout up
                vol_confirm                             # volume confirmation
            )
            
            # Short when price below daily EMA34 (bearish bias) and break below lower band
            short_condition = (
                close[i] < ema34_daily_aligned[i] and   # price below EMA34 (bearish bias)
                short_breakout and                      # Donchian breakout down
                vol_confirm                             # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or trend reverses
            if i >= 20:
                donchian_middle = (np.max(high[i-19:i+1]) + np.min(low[i-19:i+1])) / 2
                if close[i] < donchian_middle or close[i] < ema34_daily_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or trend reverses
            if i >= 20:
                donchian_middle = (np.max(high[i-19:i+1]) + np.min(low[i-19:i+1])) / 2
                if close[i] > donchian_middle or close[i] > ema34_daily_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals