#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with Weekly Pivot Direction Filter and Volume Spike Confirmation.
- Long when: Tenkan-sen crosses above Kijun-sen (TK cross bullish) AND price > Cloud (bullish) AND Weekly pivot shows bullish bias (weekly close > weekly open) AND volume > 2.0 * 20-period average volume
- Short when: Tenkan-sen crosses below Kijun-sen (TK cross bearish) AND price < Cloud (bearish) AND Weekly pivot shows bearish bias (weekly close < weekly open) AND volume > 2.0 * 20-period average volume
- Exit on opposite TK cross (exit long on bearish TK cross, exit short on bullish TK cross)
- Uses 6h primary with 1w HTF for weekly pivot bias to target 50-150 total trades over 4 years (12-37/year)
- Ichimoku provides trend/momentum/cloud support/resistance; weekly pivot filters regime; volume spike confirms breakout strength
- Designed to work in both bull (breakouts with cloud support) and bear (breakouts against cloud resistance) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Cloud (Kumo): between Senkou Span A and B
    # Bullish when price > Cloud, Bearish when price < Cloud
    # Cloud top is max(senkou_a, senkou_b), bottom is min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross signals
    # Bullish TK cross: tenkan crosses above kijun
    tk_bullish = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    # Bearish TK cross: tenkan crosses below kijun
    tk_bearish = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    # Handle first element
    tk_bullish[0] = False
    tk_bearish[0] = False
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot bias: bullish if weekly close > weekly open, bearish if weekly close < weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly pivot bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20) + 1  # Need Ichimoku (52) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish TK cross AND price above cloud AND weekly bullish bias AND volume confirmation
            if tk_bullish[i] and price_above_cloud[i] and weekly_bullish_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross AND price below cloud AND weekly bearish bias AND volume confirmation
            elif tk_bearish[i] and price_below_cloud[i] and weekly_bearish_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TK cross (opposite signal)
            if tk_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TK cross (opposite signal)
            if tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0