#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Volume_Spray_v1
Hypothesis: 
- Uses 1d Camarilla pivot levels (support/resistance) as key price levels
- Enters long near L3 support with bullish rejection, short near H3 resistance with bearish rejection
- Requires volume spike (>2x 20-period average) to confirm institutional interest
- Uses 1-week ADX > 25 to ensure we're in a trending environment (avoids chop)
- Designed for 12h timeframe to capture multi-day swings while minimizing noise
- Works in both bull/bear markets by fading extremes in trending conditions
- Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Volume_Spray_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot range
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using previous day's values)
    H3 = close_1d + 1.1 * range_1d
    H4 = close_1d + 1.5 * range_1d
    L3 = close_1d - 1.1 * range_1d
    L4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) 
        # Subsequent values using Wilder's smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_period = 14
    atr_1w = smooth_wilder(tr, atr_period)
    dm_plus_smooth = smooth_wilder(dm_plus, atr_period)
    dm_minus_smooth = smooth_wilder(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = smooth_wilder(dx, atr_period)
    
    # Align ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume filter: volume spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: only trade in trending markets (ADX > 25)
        trending = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (take profit) or trend dies
            if close[i] >= H3_aligned[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 (take profit) or trend dies
            if close[i] <= L3_aligned[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Only enter in trending markets
            if not trending:
                signals[i] = 0.0
                continue
                
            # Long entry: price near L3 support with bullish rejection + volume spike
            # Price within 0.5% of L3 and closing above it (bullish rejection)
            near_L3 = np.abs(close[i] - L3_aligned[i]) / L3_aligned[i] < 0.005
            bullish_rejection = close[i] > L3_aligned[i]
            
            # Short entry: price near H3 resistance with bearish rejection + volume spike
            near_H3 = np.abs(close[i] - H3_aligned[i]) / H3_aligned[i] < 0.005
            bearish_rejection = close[i] < H3_aligned[i]
            
            if near_L3 and bullish_rejection and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            elif near_H3 and bearish_rejection and volume_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals