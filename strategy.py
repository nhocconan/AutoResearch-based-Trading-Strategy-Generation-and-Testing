#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# - Donchian breakout: price > upper(20) or price < lower(20) on 6h timeframe
# - Weekly pivot direction: price > weekly pivot (from 1d data) for long, < for short
# - Volume confirmation: 6h volume > 1.5x 20-period average (from 1d data aligned)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Donchian provides clear breakout levels, weekly pivot gives HTF bias, volume filters weak signals
# - Works in bull markets (breakouts with HTF up bias) and bear markets (breakdowns with HTF down bias)

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for weekly pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d weekly pivot (using prior week's high, low, close)
    # For each 1d bar, calculate pivot based on prior 7 days (approximate week)
    df_1d = df_1d.copy()
    df_1d['weekly_high'] = pd.Series(df_1d['high']).rolling(window=7, min_periods=7).max().shift(1)
    df_1d['weekly_low'] = pd.Series(df_1d['low']).rolling(window=7, min_periods=7).min().shift(1)
    df_1d['weekly_close'] = pd.Series(df_1d['close']).rolling(window=7, min_periods=7).last().shift(1)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (df_1d['weekly_high'] + df_1d['weekly_low'] + df_1d['weekly_close']) / 3
    weekly_pivot_values = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_values)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels on 6h timeframe
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    for i in range(donchian_window, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_current = close[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_up = close_current > upper_channel[i]
        breakdown_down = close_current < lower_channel[i]
        
        # Weekly pivot direction from 1d data
        above_weekly_pivot = close_current > weekly_pivot_aligned[i]
        below_weekly_pivot = close_current < weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout up + above weekly pivot + volume confirmation
        if breakout_up and above_weekly_pivot and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakdown down + below weekly pivot + volume confirmation
        if breakdown_down and below_weekly_pivot and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian touch or loss of HTF bias
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price touches lower channel OR goes below weekly pivot
            exit_long = (close_current <= lower_channel[i]) or (not above_weekly_pivot)
        elif position == -1:
            # Exit short if price touches upper channel OR goes above weekly pivot
            exit_short = (close_current >= upper_channel[i]) or (not below_weekly_pivot)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals