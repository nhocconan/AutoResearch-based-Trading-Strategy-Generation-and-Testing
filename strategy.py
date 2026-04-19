#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Pivot Point Reversal with 12h Volume Confirmation and 12h Trend Filter
# - Uses daily pivot points (S1/S2 for longs, R1/R2 for shorts) as support/resistance
# - 12h volume > 1.5x 20-period average for confirmation
# - 12h EMA(34) trend filter: only take longs when price > 12h EMA34, shorts when price < 12h EMA34
# - Designed to capture reversals at key levels with trend alignment
# - Target: 25-40 trades/year to minimize fee drag while capturing meaningful moves

name = "4h_PivotPoint_12hVolume_12hTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume and trend filters
    df_12h = get_htf_data(prices, '12h')
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P-H, S2 = P-(H-L), R1 = 2P-L, R2 = P+(H-L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d
    s2 = pivot - (high_1d - low_1d)
    r1 = 2 * pivot - low_1d
    r2 = pivot + (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 12h average volume (scaled)
        # Scale 12h average to 4h: 12h has 3x 4h bars, so divide by 3
        volume_filter = vol_ma_12h_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_12h_aligned[i] / 3.0)
        
        if position == 0:
            # Look for long entry: price at or below S1/S2 + uptrend (price > 12h EMA34) + volume
            if (close[i] <= s1_aligned[i] or close[i] <= s2_aligned[i]) and close[i] > ema_34_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price at or above R1/R2 + downtrend (price < 12h EMA34) + volume
            elif (close[i] >= r1_aligned[i] or close[i] >= r2_aligned[i]) and close[i] < ema_34_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price reaches pivot or trend reverses
            if close[i] >= pivot_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price reaches pivot or trend reverses
            if close[i] <= pivot_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals