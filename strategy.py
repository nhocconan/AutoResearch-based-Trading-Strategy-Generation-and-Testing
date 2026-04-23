#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud Breakout with Weekly Trend Filter and Volume Confirmation
- Uses Ichimoku cloud (Senkou Span A/B) from daily timeframe for dynamic support/resistance
- TK Cross (Tenkan/Kijun) from 6h for entry timing with cloud filter
- Weekly EMA20 defines higher timeframe trend: only trade in direction of weekly trend
- Volume confirmation (> 1.8x 24-period average) filters weak signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by combining momentum (TK cross) with trend and structure filters
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
    
    # Calculate daily Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 24)  # for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud AND above weekly EMA20 AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and  # TK cross bullish
                close[i] > cloud_top and                  # Price above cloud
                close[i] > ema_20_1w_aligned[i] and       # Above weekly trend
                volume[i] > 1.8 * vol_ma[i]):             # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud AND below weekly EMA20 AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and  # TK cross bearish
                  close[i] < cloud_bottom and               # Price below cloud
                  close[i] < ema_20_1w_aligned[i] and       # Below weekly trend
                  volume[i] > 1.8 * vol_ma[i]):             # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit: TK cross reverses OR price re-enters cloud
            exit_signal = False
            
            if position == 1:
                # Exit long when TK cross bearish OR price drops below cloud top
                if (tenkan_aligned[i] < kijun_aligned[i] or close[i] < cloud_top):
                    exit_signal = True
            elif position == -1:
                # Exit short when TK cross bullish OR price rises above cloud bottom
                if (tenkan_aligned[i] > kijun_aligned[i] or close[i] > cloud_bottom):
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