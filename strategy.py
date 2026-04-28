#!/usr/bin/env python3
# Hypothesis: 12-hour Camarilla R1/S1 breakout with daily ADX trend filter and volume confirmation.
# Uses 12h timeframe to capture multi-day moves while avoiding excessive trading.
# Long when price breaks above R1 with ADX>25 and volume surge.
# Short when price breaks below S1 with ADX>25 and volume surge.
# Exit when price reverts to pivot or trend weakens.
# Designed for 12h to target 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear markets by requiring strong trend (ADX>25) and volume confirmation.

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
    
    # Get daily data for pivot points and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Pivot = (H+L+C)/3
    # We use previous day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = high_1d[0]  # First day uses same day
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Calculate pivot and levels
    pivot = (high_prev + low_prev + close_prev) / 3
    camarilla_range = (high_prev - low_prev) * 1.1 / 12
    r1 = pivot + camarilla_range
    s1 = pivot - camarilla_range
    
    # Calculate daily ADX (14-period) for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di_smoothed = wilders_smoothing(plus_dm, 14)
    minus_di_smoothed = wilders_smoothing(minus_dm, 14)
    
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align daily data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 2.0x 24-period average (2 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        # Reversion conditions (exit when price returns to pivot area)
        # We'll exit when price crosses back below R1 for longs or above S1 for shorts
        # But also add a time-based exit or pivot reversion
        
        # Entry conditions with volume confirmation
        long_entry = strong_trend and breakout_long and volume_filter[i]
        short_entry = strong_trend and breakout_short and volume_filter[i]
        
        # Exit conditions
        # Exit long when price falls below R1 or trend weakens
        long_exit = (not strong_trend) or (close[i] < r1_aligned[i]) or (position == 1 and close[i] < (r1_aligned[i] + s1_aligned[i])/2)
        # Exit short when price rises above S1 or trend weakens
        short_exit = (not strong_trend) or (close[i] > s1_aligned[i]) or (position == -1 and close[i] > (r1_aligned[i] + s1_aligned[i])/2)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dADX_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0