#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud Breakout with Weekly EMA20 Trend and Volume Confirmation
- Uses Ichimoku components (Tenkan, Kijun, Senkou Span A/B) from 6h for cloud breakout signals
- Weekly EMA20 defines higher timeframe trend: only take breakouts in direction of weekly trend
- Volume confirmation (> 2.0x 20-period average) ensures institutional participation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by aligning with weekly trend while using 6h Ichimoku for precise entries
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
    
    # Calculate Ichimoku components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud boundaries (shifted back 26 periods for alignment)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52 + 26, 20)  # for Ichimoku (52+26) and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_shifted[i], senkou_b_shifted[i])
        cloud_bottom = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Long: Price breaks above cloud AND Tenkan > Kijun AND price above weekly EMA20 AND volume spike
            if (close[i] > cloud_top and 
                tenkan_sen[i] > kijun_sen[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND Tenkan < Kijun AND price below weekly EMA20 AND volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to cloud OR trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price closes below cloud OR weekly trend turns bearish
                if (close[i] < cloud_top or close[i] < ema_20_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price closes above cloud OR weekly trend turns bullish
                if (close[i] > cloud_bottom or close[i] > ema_20_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyEMA20_Trend_Volume"
timeframe = "6h"
leverage = 1.0