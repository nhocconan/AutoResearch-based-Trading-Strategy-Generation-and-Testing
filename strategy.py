#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud breakout with 12h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Uses 1d Ichimoku for multi-timeframe alignment. Designed for BTC/ETH robustness: Ichimoku provides dynamic support/resistance, EMA50 ensures trend alignment, volume confirms momentum. Targets 12-35 trades/year on 6h timeframe with discrete position sizing to minimize fee drag.

name = "6h_Ichimoku_Cloud_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after lookback for Senkou Span B
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or 
            np.isnan(senkou_span_b[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:
            # LONG: Price breaks above cloud, Tenkan > Kijun (bullish TK cross), price > 12h EMA50, volume spike
            if (close[i] > cloud_top and 
                tenkan_sen[i] > kijun_sen[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud, Tenkan < Kijun (bearish TK cross), price < 12h EMA50, volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below cloud OR Tenkan < Kijun (TK cross bearish)
            if (close[i] < cloud_bottom or 
                tenkan_sen[i] < kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above cloud OR Tenkan > Kijun (TK cross bullish)
            if (close[i] > cloud_top or 
                tenkan_sen[i] > kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals