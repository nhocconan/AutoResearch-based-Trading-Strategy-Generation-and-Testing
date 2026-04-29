#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above 6h Donchian upper channel, weekly pivot > previous weekly pivot (bullish bias), volume > 1.5x average
# Short when price breaks below 6h Donchian lower channel, weekly pivot < previous weekly pivot (bearish bias), volume > 1.5x average
# Exit when price reverts to 6h Donchian midpoint (mean reversion)
# Uses 1w for trend bias (novel: weekly pivot slope), 6h only for entry timing and breakout levels
# Target: 50-150 total trades over 4 years = 12-37/year. Discrete sizing 0.25 to limit fee drag.

name = "6h_Donchian20_1wPivotSlope_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data for Donchian channels (based on previous 20 periods)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 21:  # Need 20 for Donchian + 1 for previous
        return np.zeros(n)
    
    # Calculate 6h Donchian channels using previous 20 completed 6h bars
    # Shift by 1 to use completed periods only (no look-ahead)
    prev_high = df_6h['high'].shift(1).values
    prev_low = df_6h['low'].shift(1).values
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 6h indicators to 6h timeframe (no additional delay needed for Donchian)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    
    # Get 1w data for weekly pivot slope filter (novel: weekly pivot direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_values = weekly_pivot.values
    
    # Weekly pivot slope: current pivot > previous pivot = bullish bias
    # Shift by 1 to use completed weeks only (no look-ahead)
    prev_weekly_pivot = np.roll(weekly_pivot_values, 1)
    prev_weekly_pivot[0] = np.nan  # First value has no previous
    weekly_pivot_bullish = weekly_pivot_values > prev_weekly_pivot
    weekly_pivot_bearish = weekly_pivot_values < prev_weekly_pivot
    
    # Align 1w indicators to 6h timeframe with additional delay for pivot confirmation
    # Weekly pivot needs 1 extra bar confirmation (wait for next weekly candle to open)
    weekly_pivot_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_bullish.astype(float), additional_delay_bars=1)
    weekly_pivot_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_bearish.astype(float), additional_delay_bars=1)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_pivot_bullish_aligned[i]) or 
            np.isnan(weekly_pivot_bearish_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dch_high = donchian_high_aligned[i]
        curr_dch_low = donchian_low_aligned[i]
        curr_dch_mid = donchian_mid_aligned[i]
        curr_wk_pivot_bull = weekly_pivot_bullish_aligned[i]
        curr_wk_pivot_bear = weekly_pivot_bearish_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 6h Donchian midpoint (mean reversion)
            if curr_close < curr_dch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 6h Donchian midpoint (mean reversion)
            if curr_close > curr_dch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average (balanced filter)
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above 6h Donchian upper channel, weekly pivot bullish, volume confirmed
            if curr_high > curr_dch_high and curr_wk_pivot_bull > 0.5 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 6h Donchian lower channel, weekly pivot bearish, volume confirmed
            elif curr_low < curr_dch_low and curr_wk_pivot_bear > 0.5 and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals