#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND weekly pivot direction is bullish AND volume > 1.5x average.
# Short when price breaks below Donchian(20) lower band AND weekly pivot direction is bearish AND volume > 1.5x average.
# Exit when price crosses back below Donchian middle (for long) or above Donchian middle (for short).
# Weekly pivot direction based on price relative to weekly pivot point (PP) from prior week.
# Uses Donchian breakouts for trend capture with weekly pivot filter to align with higher timeframe bias.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "6h_Donchian_1wPivot_Volume"
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
    
    # 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Weekly high/low/close: resample daily to weekly using last values in week
    # We'll compute pivot for each day based on prior week's H/L/C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for weekly pivot components
    weekly_high = np.full_like(high_1d, np.nan)
    weekly_low = np.full_like(low_1d, np.nan)
    weekly_close = np.full_like(close_1d, np.nan)
    
    # Compute weekly aggregates: for each day, use the last values in the week (assuming week starts Monday)
    # Simpler: use expanding window with weekly reset - but we'll use a rolling window of 5 days (approx week)
    # More accurate: group by week number
    # Since we don't have explicit dates, approximate with 5-day rolling (trading week)
    window = 5
    weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly pivot direction: bullish if close > PP, bearish if close < PP
    # We'll use the prior week's PP to avoid look-ahead
    weekly_pp_lag = np.roll(weekly_pp, 1)  # Use prior week's PP
    weekly_pp_lag[0] = np.nan
    
    # Align weekly data to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp_lag)
    pp_direction = np.where(close_1d > weekly_pp_lag, 1, -1)  # 1=bullish, -1=bearish
    pp_direction[np.isnan(weekly_pp_lag)] = 0
    pp_direction_aligned = align_htf_to_ltf(prices, df_1d, pp_direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(pp_direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band, bullish weekly pivot, volume spike
            long_cond = (close[i] > donchian_upper[i]) and (pp_direction_aligned[i] == 1) and volume_filter[i]
            # Short conditions: break below lower band, bearish weekly pivot, volume spike
            short_cond = (close[i] < donchian_lower[i]) and (pp_direction_aligned[i] == -1) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below middle band
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above middle band
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals