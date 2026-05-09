#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud with weekly trend filter and volume confirmation
# Long when price is above Kumo cloud, Tenkan > Kijun, and weekly trend is bullish (price above weekly Kumo)
# Short when price is below Kumo cloud, Tenkan < Kijun, and weekly trend is bearish (price below weekly Kumo)
# Exit when Tenkan and Kijun cross in opposite direction
# Uses Ichimoku for trend/momentum, weekly timeframe for trend filter, volume for confirmation
# Designed for 6h timeframe with moderate trade frequency and trend-following in both bull/bear markets

name = "6h_Ichimoku_Cloud_WeeklyTrend_Volume"
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2).values
    
    # Weekly trend filter: price above/below weekly Kumo
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # Weekly Tenkan and Kijun
    w_period9_high = pd.Series(whigh).rolling(window=9, min_periods=9).max()
    w_period9_low = pd.Series(wlow).rolling(window=9, min_periods=9).min()
    w_tenkan = ((w_period9_high + w_period9_low) / 2).values
    
    w_period26_high = pd.Series(whigh).rolling(window=26, min_periods=26).max()
    w_period26_low = pd.Series(wlow).rolling(window=26, min_periods=26).min()
    w_kijun = ((w_period26_high + w_period26_low) / 2).values
    
    # Weekly Senkou Span A and B
    w_senkou_a = ((w_tenkan + w_kijun) / 2)
    w_period52_high = pd.Series(whigh).rolling(window=52, min_periods=52).max()
    w_period52_low = pd.Series(wlow).rolling(window=52, min_periods=52).min()
    w_senkou_b = ((w_period52_high + w_period52_low) / 2).values
    
    # Weekly Kumo (cloud) boundaries
    w_kumo_top = np.maximum(w_senkou_a, w_senkou_b)
    w_kumo_bottom = np.minimum(w_senkou_a, w_senkou_b)
    
    # Align weekly Ichimoku components to 6h timeframe
    w_kumo_top_aligned = align_htf_to_ltf(prices, df_1w, w_kumo_top)
    w_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, w_kumo_bottom)
    
    # Volume confirmation: current volume > 1.5x 26-period average
    vol_ma = pd.Series(volume).rolling(window=26, min_periods=26).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(w_kumo_top_aligned[i]) or np.isnan(w_kumo_bottom_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current Kumo (cloud) boundaries
        kumo_top = np.maximum(senkou_a[i], senkou_b[i])
        kumo_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Enter long: price above Kumo, Tenkan > Kijun, weekly trend bullish, volume confirmation
            if (close[i] > kumo_top and 
                tenkan[i] > kijun[i] and
                close[i] > w_kumo_top_aligned[i] and  # price above weekly Kumo (bullish trend)
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below Kumo, Tenkan < Kijun, weekly trend bearish, volume confirmation
            elif (close[i] < kumo_bottom and 
                  tenkan[i] < kijun[i] and
                  close[i] < w_kumo_bottom_aligned[i] and  # price below weekly Kumo (bearish trend)
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun (trend weakness)
            if tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun (trend weakness)
            if tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals