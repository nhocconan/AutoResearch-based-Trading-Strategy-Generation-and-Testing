#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 123 Reversal Pattern with 1d ADX trend filter and volume confirmation
# The 123 reversal pattern identifies trend exhaustion and potential reversal points:
# Point 1: Swing high/low, Point 2: Pullback, Point 3: Failed retest of Point 1
# In Point 3, we look for failure to exceed Point 1 with weakening momentum
# ADX filter ensures we only trade in trending conditions (ADX > 25)
# Volume confirmation ensures institutional participation
# Works in both bull and bear markets by identifying reversal points in trends
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_123Reversal_1dADX_Volume"
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
    
    # 1d ADX for trend filter (only trade when ADX > 25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First element is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])  # Skip first NaN in TR
        # Wilder's smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr > 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = smoothed_avg(dx, 14)
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Swing points for 123 pattern (using 5-bar lookback)
    def find_swing_high(arr, lookback=2):
        swings = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.max(arr[i-lookback:i+lookback+1]):
                swings[i] = arr[i]
        return swings
    
    def find_swing_low(arr, lookback=2):
        swings = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.min(arr[i-lookback:i+lookback+1]):
                swings[i] = arr[i]
        return swings
    
    swing_high = find_swing_high(high, 2)
    swing_low = find_swing_low(low, 2)
    
    # Volume confirmation: volume > 1.2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(swing_high[i]) or 
            np.isnan(swing_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for 123 pattern completion
            # Bullish 123: Point 1 (swing low), Point 2 (pullback high), Point 3 (failed retest of Point 1 low)
            if not np.isnan(swing_low[i]):
                point1_low = swing_low[i]
                # Find Point 2: pullback high after Point 1
                point2_high = np.nan
                point1_idx = i
                # Look back for the swing low
                for j in range(max(0, i-20), i):
                    if not np.isnan(swing_low[j]) and swing_low[j] == point1_low:
                        point1_idx = j
                        # Find pullback high between Point 1 and current
                        pullback_high = np.max(high[point1_idx:i+1])
                        point2_high = pullback_high
                        break
                
                if not np.isnan(point2_high) and point2_high > point1_low:
                    # Point 3: current price fails to break above Point 2 but holds above Point 1
                    # AND shows rejection (close near low)
                    if (close[i] < point2_high and 
                        close[i] > point1_low and
                        close[i] <= np.percentile(high[max(0,i-4):i+1], 30) and  # Close in lower 30% of recent range
                        adx_1d_aligned[i] > 25 and  # Trending market
                        volume_confirm[i]):
                        signals[i] = 0.25
                        position = 1
            
            # Bearish 123: Point 1 (swing high), Point 2 (pullback low), Point 3 (failed retest of Point 1 high)
            if not np.isnan(swing_high[i]):
                point1_high = swing_high[i]
                # Find Point 2: pullback low after Point 1
                point2_low = np.nan
                point1_idx = i
                # Look back for the swing high
                for j in range(max(0, i-20), i):
                    if not np.isnan(swing_high[j]) and swing_high[j] == point1_high:
                        point1_idx = j
                        # Find pullback low between Point 1 and current
                        pullback_low = np.min(low[point1_idx:i+1])
                        point2_low = pullback_low
                        break
                
                if not np.isnan(point2_low) and point2_low < point1_high:
                    # Point 3: current price fails to break below Point 2 but holds below Point 1
                    # AND shows rejection (close near high)
                    if (close[i] > point2_low and 
                        close[i] < point1_high and
                        close[i] >= np.percentile(low[max(0,i-4):i+1], 70) and  # Close in upper 30% of recent range
                        adx_1d_aligned[i] > 25 and  # Trending market
                        volume_confirm[i]):
                        signals[i] = -0.25
                        position = -1
                        
        elif position == 1:
            # Long: exit if price breaks below Point 1 low or ADX weakens
            if not np.isnan(swing_low[i]):
                point1_low = swing_low[i]
                # Find the most recent swing low
                for j in range(i, max(0, i-20), -1):
                    if not np.isnan(swing_low[j]):
                        point1_low = swing_low[j]
                        break
                if close[i] < point1_low or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Point 1 high or ADX weakens
            if not np.isnan(swing_high[i]):
                point1_high = swing_high[i]
                # Find the most recent swing high
                for j in range(i, max(0, i-20), -1):
                    if not np.isnan(swing_high[j]):
                        point1_high = swing_high[j]
                        break
                if close[i] > point1_high or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals