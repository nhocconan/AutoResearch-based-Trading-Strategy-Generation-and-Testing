#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud with weekly trend filter and volume confirmation.
# Long when Tenkan > Kijun, price above cloud, weekly trend up (price > weekly EMA200), and volume > 1.5x average.
# Short when Tenkan < Kijun, price below cloud, weekly trend down (price < weekly EMA200), and volume > 1.5x average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Ichimoku provides institutional-grade trend/mean reversion signals; weekly EMA200 filters counter-trend noise in bear markets.
# Volume confirmation ensures momentum validity. Designed for low trade frequency to avoid fee drag.

name = "6h_Ichimoku_WeeklyEMA200_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    
    tenkan_sen = ((period9_high + period9_low) / 2).values
    kijun_sen = ((period26_high + period26_low) / 2).values
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Shift senkou spans forward by 26 periods (cloud ahead)
    senkou_span_a = senkou_span_a.shift(26).values
    senkou_span_b = senkou_span_b.shift(26).values
    
    # Calculate average volume for confirmation (26-period)
    avg_volume = pd.Series(volume).rolling(window=26, min_periods=26).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after sufficient data for Ichimoku
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:
            # LONG: Tenkan > Kijun, price above cloud, weekly trend up, volume spike
            if (tenkan_sen[i] > kijun_sen[i] and 
                close[i] > cloud_top and 
                close[i] > ema_200_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan < Kijun, price below cloud, weekly trend down, volume spike
            elif (tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan < Kijun OR price below cloud
            if (tenkan_sen[i] < kijun_sen[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan > Kijun OR price above cloud
            if (tenkan_sen[i] > kijun_sen[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals