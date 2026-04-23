#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d Weekly Pivot Direction Filter and Volume Confirmation
- Uses Ichimoku (Tenkan/Kijun/Senkou) on 6h for trend and momentum
- 1d weekly pivot (R1/S1) provides structural bias: long only above weekly pivot, short only below
- Volume confirmation (>1.5x 20-period average) ensures institutional participation
- Entry: Tenkan-Kijun cross in direction of weekly pivot bias with volume confirmation
- Exit: Opposite Tenkan-Kijun cross or price crosses Kijun (dynamic support/resistance)
- Designed for 6h timeframe to capture medium-term swings in both bull and bear markets
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

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
    
    # Get 1d data for weekly pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Using prior complete week: need at least 5 trading days
    if len(df_1d) < 5:
        weekly_pivot = np.full(len(prices), np.nan)
        r1 = np.full(len(prices), np.nan)
        s1 = np.full(len(prices), np.nan)
    else:
        # Resample to weekly using actual weekly data (not synthetic)
        weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
        weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week low
        weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week close
        
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
        r1 = 2 * weekly_pivot - weekly_low
        s1 = 2 * weekly_pivot - weekly_high
        
        # Align to 6h timeframe
        weekly_pivot = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
        r1 = align_htf_to_ltf(prices, df_1d, r1.values)
        s1 = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(26, 52, 20, 5)  # Need 26 for Kijun, 52 for Senkou B, 20 for volume, 5 for weekly
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(weekly_pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku signals
        tenkan_cross_above_kijun = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tenkan_cross_below_kijun = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price relative to cloud (simplified: using Senkou A/B)
        above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
        below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
        
        # Weekly pivot bias
        above_weekly_pivot = close[i] > weekly_pivot[i]
        below_weekly_pivot = close[i] < weekly_pivot[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND above weekly pivot AND volume confirmation
            if tenkan_cross_above_kijun and above_weekly_pivot and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND below weekly pivot AND volume confirmation
            elif tenkan_cross_below_kijun and below_weekly_pivot and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Tenkan crosses below Kijun OR price drops below Kijun
                if tenkan_cross_below_kijun or close[i] < kijun[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Tenkan crosses above Kijun OR price rises above Kijun
                if tenkan_cross_above_kijun or close[i] > kijun[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_1dWeeklyPivot_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0