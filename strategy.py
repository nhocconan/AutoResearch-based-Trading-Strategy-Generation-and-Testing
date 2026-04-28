# 12h_PivotPoint_Reversal_With_Trend_and_Volume
# Hypothesis: Price tends to reverse at key support/resistance (pivot levels) when aligned with higher timeframe trend and volume confirmation.
# This strategy works in both bull and bear markets by taking reversals at institutional pivot points, using 1-day pivot levels calculated from prior day's OHLC.
# Trend filter ensures we trade in direction of higher timeframe momentum, reducing false signals.
# Volume confirmation adds conviction to reversals.
# Timeframe: 12h (low frequency to minimize fee drag, target 50-150 trades over 4 years)
# Pivot levels: Calculated from prior 1-day OHLC (standard floor trader pivots)
# Entry: Long when price crosses above S1 with bullish trend and volume spike; Short when price crosses below R1 with bearish trend and volume spike
# Exit: Opposite pivot level touch or trend reversal
# Position sizing: 0.25 (discrete to minimize churn)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for pivot calculation and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day's OHLC
    # Standard formula: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot, R1, S1 for each day
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe (use prior day's levels for current day)
    # Shift by 1 to use previous day's pivot levels (avoid look-ahead)
    pivot_shifted = np.roll(pivot, 1)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    # First day has no prior day, set to NaN
    pivot_shifted[0] = np.nan
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_shifted)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_shifted)
    
    # 1-day EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = (close[i] > s1_aligned[i]) and trend_up and volume_spike[i]
        short_entry = (close[i] < r1_aligned[i]) and trend_down and volume_spike[i]
        
        # Exit conditions: touch opposite pivot level or trend reversal
        long_exit = (close[i] < pivot_aligned[i]) or not trend_up
        short_exit = (close[i] > pivot_aligned[i]) or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_PivotPoint_Reversal_With_Trend_and_Volume"
timeframe = "12h"
leverage = 1.0