#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above 6h Donchian upper band AND weekly pivot > previous weekly pivot (bullish bias) AND 6h volume > 1.5x 20-period volume MA.
# Short when price breaks below 6h Donchian lower band AND weekly pivot < previous weekly pivot (bearish bias) AND 6h volume > 1.5x 20-period volume MA.
# Exit on retracement to 6h Donchian middle band or pivot bias reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Donchian provides clear breakout levels, weekly pivot confirms higher-timeframe structure, volume ensures participation.
# Works in both bull and bear markets by only trading breakouts aligned with weekly pivot momentum.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_prev = np.roll(weekly_pivot, 1)  # Previous week's pivot
    weekly_pivot_prev[0] = np.nan  # First value has no previous
    
    # Align weekly pivot to 6h timeframe (wait for weekly bar to close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_pivot_prev_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_prev)
    
    # Calculate 6h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_rolling_max
    donchian_lower = low_rolling_min
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_pivot_prev_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = close_val > donchian_upper[i]   # Price closes above upper band
        breakout_down = close_val < donchian_lower[i]  # Price closes below lower band
        
        # Weekly pivot bias conditions
        pivot_bullish = weekly_pivot_aligned[i] > weekly_pivot_prev_aligned[i]  # Rising pivot = bullish bias
        pivot_bearish = weekly_pivot_aligned[i] < weekly_pivot_prev_aligned[i]  # Falling pivot = bearish bias
        
        if position == 0:
            # Long: Donchian breakout up AND bullish pivot bias AND volume spike AND session
            if breakout_up and pivot_bullish and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND bearish pivot bias AND volume spike AND session
            elif breakout_down and pivot_bearish and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian middle band OR pivot bias turns bearish
            if close_val < donchian_middle[i] or not pivot_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian middle band OR pivot bias turns bullish
            if close_val > donchian_middle[i] or not pivot_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals