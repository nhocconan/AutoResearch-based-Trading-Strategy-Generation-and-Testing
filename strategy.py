#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_WeeklyTrend
Hypothesis: Ichimoku TK cross with daily cloud filter and weekly trend alignment on 6h timeframe.
Ichimoku provides objective trend/momentum signals while higher timeframe filters prevent counter-trend trades.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.0, ±0.25).
Works in bull/bear via weekly trend filter and cloud support/resistance.
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
    
    # Load 1d data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for entry but for cloud thickness if needed
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top/bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (max of Ichimoku periods + weekly EMA warmup)
    start_idx = max(52, 50) + 26  # 52 for Senkou B, 50 for weekly EMA, +26 for Chikou shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: TK cross bullish + price above cloud + weekly uptrend
        if (tenkan_aligned[i] > kijun_aligned[i] and  # TK cross bullish
            close[i] > cloud_top[i] and               # Price above cloud
            close[i] > ema_50_1w_aligned[i]):         # Price above weekly EMA50 (uptrend)
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross bearish + price below cloud + weekly downtrend
        elif (tenkan_aligned[i] < kijun_aligned[i] and  # TK cross bearish
              close[i] < cloud_bottom[i] and            # Price below cloud
              close[i] < ema_50_1w_aligned[i]):         # Price below weekly EMA50 (downtrend)
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross in opposite direction OR price crosses weekly EMA50
        elif position == 1 and (tenkan_aligned[i] < kijun_aligned[i] or 
                                close[i] < ema_50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tenkan_aligned[i] > kijun_aligned[i] or 
                                 close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_WeeklyTrend"
timeframe = "6h"
leverage = 1.0