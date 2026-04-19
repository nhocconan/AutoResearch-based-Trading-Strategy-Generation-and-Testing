#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Pivot-based strategy with volume confirmation and 1-day trend filter
# - Long when price breaks above 1-day pivot resistance (R1) with volume spike (>1.5x 20-period 12h avg volume) and price > 1-day EMA50 (uptrend)
# - Short when price breaks below 1-day pivot support (S1) with volume spike and price < 1-day EMA50 (downtrend)
# - Exit when price returns to 1-day pivot point or trend reverses
# - Uses 12h timeframe to reduce trade frequency and focus on significant breakouts
# - Designed for both bull and bear markets by following the 1-day trend filter
# - Target: 12-30 trades/year (48-120 total over 4 years) to minimize fee drag

name = "12h_Pivot_R1S1_Breakout_Volume_1dTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot points from previous 1d bar
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point calculations
    pivot_point = (prev_high + prev_low + prev_close) / 3
    pivot_range = prev_high - prev_low
    pivot_r1 = pivot_point + pivot_range  # R1 = P + (H - L)
    pivot_s1 = pivot_point - pivot_range  # S1 = P - (H - L)
    
    # Align pivot levels to 12h timeframe (wait for 1d bar to close)
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1d, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1d, pivot_s1)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 12h volume > 1.5x 20-period average volume (using 12h data directly)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_r1_aligned[i]) or np.isnan(pivot_s1_aligned[i]) or 
            np.isnan(pivot_point_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 20-period average volume
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: price breaks above R1 + volume spike + uptrend
            if (close[i] > pivot_r1_aligned[i] and 
                close[i-1] <= pivot_r1_aligned[i-1] and  # Just broke above
                volume_filter and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S1 + volume spike + downtrend
            elif (close[i] < pivot_s1_aligned[i] and 
                  close[i-1] >= pivot_s1_aligned[i-1] and  # Just broke below
                  volume_filter and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot point or trend reverses
            if (close[i] <= pivot_point_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot point or trend reverses
            if (close[i] >= pivot_point_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals