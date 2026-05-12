#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D DATA FOR ICHIMOKU CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku Components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # === 1D TREND FILTER: EMA50 ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Need 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Green cloud: Senkou A > Senkou B (bullish)
        # Red cloud: Senkou A < Senkou B (bearish)
        green_cloud = senkou_a_6h[i] > senkou_b_6h[i]
        red_cloud = senkou_a_6h[i] < senkou_b_6h[i]
        
        if position == 0:
            # LONG: Price above cloud, Tenkan > Kijun, above 1d EMA50, volume spike
            if (close[i] > senkou_a_6h[i] and close[i] > senkou_b_6h[i] and  # Above cloud
                tenkan_6h[i] > kijun_6h[i] and                             # Bullish TK cross
                close[i] > ema50_1d_aligned[i] and                         # Above 1d EMA50
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud, Tenkan < Kijun, below 1d EMA50, volume spike
            elif (close[i] < senkou_a_6h[i] and close[i] < senkou_b_6h[i] and  # Below cloud
                  tenkan_6h[i] < kijun_6h[i] and                             # Bearish TK cross
                  close[i] < ema50_1d_aligned[i] and                         # Below 1d EMA50
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below cloud OR TK cross turns bearish
            if (close[i] < senkou_a_6h[i] or close[i] < senkou_b_6h[i] or  # Below cloud
                tenkan_6h[i] < kijun_6h[i]):                              # Bearish TK cross
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above cloud OR TK cross turns bullish
            if (close[i] > senkou_a_6h[i] or close[i] > senkou_b_6h[i] or  # Above cloud
                tenkan_6h[i] > kijun_6h[i]):                              # Bullish TK cross
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals