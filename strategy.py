#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_EMA60_VolumeSurge"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (novel approach - not overused)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Standard pivot point calculation: P = (H + L + C)/3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    # Support 2: S2 = P - (H - L)
    # Resistance 2: R2 = P + (H - L)
    
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly resistance levels
    weekly_r1 = (2 * weekly_pivot) - prev_week_high
    weekly_r2 = weekly_pivot + (prev_week_high - prev_week_low)
    # Weekly support levels
    weekly_s1 = (2 * weekly_pivot) - prev_week_low
    weekly_s2 = weekly_pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Trend filter: 60-period EMA on daily timeframe
    ema60_1d = pd.Series(df_1d['close']).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema60_1d_6h = align_htf_to_ltf(prices, df_1d, ema60_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0 = flat, 1 = long, -1 = short
    
    start_idx = max(60, 20)  # Need enough data for EMA60 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_r2_6h[i]) or np.isnan(weekly_s1_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or np.isnan(ema60_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current values
        pp = weekly_pivot_6h[i]
        r1 = weekly_r1_6h[i]
        r2 = weekly_r2_6h[i]
        s1 = weekly_s1_6h[i]
        s2 = weekly_s2_6h[i]
        ema60 = ema60_1d_6h[i]
        vol_surge = volume_surge[i]
        
        if position == 0:
            # Look for long setup: price above weekly pivot and EMA60 with volume surge
            if close[i] > pp and close[i] > ema60 and vol_surge:
                # Stronger signal if breaking above R1
                if close[i] > r1:
                    signals[i] = 0.30  # Full position
                else:
                    signals[i] = 0.20  # Half position
                position = 1
            # Look for short setup: price below weekly pivot and EMA60 with volume surge
            elif close[i] < pp and close[i] < ema60 and vol_surge:
                # Stronger signal if breaking below S1
                if close[i] < s1:
                    signals[i] = -0.30  # Full position
                else:
                    signals[i] = -0.20  # Half position
                position = -1
        
        elif position == 1:
            # Long position management
            # Exit if price falls below weekly pivot (trend change)
            if close[i] < pp:
                signals[i] = 0.0
                position = 0
            # Optional: take profit at R2
            elif close[i] >= r2:
                signals[i] = 0.10  # Scale down to 1/3 position
            else:
                signals[i] = 0.20  # Maintain position
        
        elif position == -1:
            # Short position management
            # Exit if price rises above weekly pivot (trend change)
            if close[i] > pp:
                signals[i] = 0.0
                position = 0
            # Optional: take profit at S2
            elif close[i] <= s2:
                signals[i] = -0.10  # Scale down to 1/3 position
            else:
                signals[i] = -0.20  # Maintain position
    
    return signals