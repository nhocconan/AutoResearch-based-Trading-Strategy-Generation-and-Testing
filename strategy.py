#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
- Uses Ichimoku Cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h for trend direction and momentum
- 1d EMA50 filter: only trade in direction of daily trend (avoid counter-trend whipsaws)
- Volume confirmation (> 1.5x 20-period average) ensures breakout strength
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Ichimoku provides built-in support/resistance (cloud) and momentum (TK cross)
- Works in both bull and bear markets by trading with higher timeframe trend
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
    
    # Calculate 6h Ichimoku components
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Calculate 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 50, 20)  # Ichimoku, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Ichimoku signals
        # Bullish: price above cloud AND Tenkan > Kijun
        price_above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # 1d trend filter
        uptrend_1d = close[i] > ema_50_aligned[i]
        downtrend_1d = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: bullish Ichimoku + uptrend on 1d + volume spike
            long_signal = (price_above_cloud and 
                          tk_bullish and
                          uptrend_1d and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: bearish Ichimoku + downtrend on 1d + volume spike
            short_signal = (price_below_cloud and 
                           tk_bearish and
                           downtrend_1d and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Ichimoku signal or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish Ichimoku or trend turns down
                if (price_below_cloud or tk_bearish or not uptrend_1d):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish Ichimoku or trend turns up
                if (price_above_cloud or tk_bullish or not downtrend_1d):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0