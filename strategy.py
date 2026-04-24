#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + 1d Weekly Pivot Direction + Volume Confirmation.
- Ichimoku (Tenkan/Kijun cross + price vs cloud) from 6h provides trend/momentum signals.
- 1d weekly pivot (using Mon-Fri weekly OHLC) determines higher-timeframe bias: only take longs above weekly pivot, shorts below.
- Volume spike (>1.5x 20-period average) confirms signal validity.
- Discrete position sizing (0.25) balances return and fee drag.
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
- Works in bull/bear via weekly pivot filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot from 1d data (Mon-Fri weekly OHLC)
    if len(df_1d) >= 5:
        # Convert to DataFrame for easy resampling to weekly
        df_1d_df = pd.DataFrame({
            'open': df_1d['open'],
            'high': df_1d['high'],
            'low': df_1d['low'],
            'close': df_1d['close']
        }, index=pd.to_datetime(df_1d.index))
        # Resample to weekly (Monday start)
        weekly = df_1d_df.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        if len(weekly) >= 2:
            weekly_high = weekly['high'].values
            weekly_low = weekly['low'].values
            weekly_close = weekly['close'].values
            # Weekly pivot = (H + L + C) / 3
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
            # Align weekly pivot to 6h timeframe (using previous completed weekly bar)
            weekly_pivot_aligned = align_htf_to_ltf(prices, weekly, weekly_pivot)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # Ichimoku components on 6h (primary timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(26, 52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Ichimoku signal
        # Bullish: price above cloud, Tenkan > Kijun
        # Bearish: price below cloud, Tenkan < Kijun
        # Cloud top/bottom
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: bullish Ichimoku + price above weekly pivot + volume spike
            if (close[i] > cloud_top and tenkan[i] > kijun[i] and 
                close[i] > weekly_pivot_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish Ichimoku + price below weekly pivot + volume spike
            elif (close[i] < cloud_bottom and tenkan[i] < kijun[i] and 
                  close[i] < weekly_pivot_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below cloud OR Tenkan < Kijun
            if close[i] < cloud_bottom or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above cloud OR Tenkan > Kijun
            if close[i] > cloud_top or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0