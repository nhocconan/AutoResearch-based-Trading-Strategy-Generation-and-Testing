#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_Volume
Hypothesis: Uses Ichimoku Cloud from 1d timeframe as trend filter and support/resistance, with Tenkan/Kijun cross on 6h for entry timing and volume confirmation. 
In bull markets: price above cloud, TK cross up, volume spike → long.
In bear markets: price below cloud, TK cross down, volume spike → short.
Exits when price crosses opposite Kumo edge or TK reverse cross.
Ichimoku provides multi-layer support/resistance and trend direction, reducing false breakouts.
Target: 20-40 trades/year via strict entry requiring trend alignment, momentum cross, and volume.
"""

name = "6h_Ichimoku_Cloud_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Ichimoku ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Ichimoku to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_a_6h, senkou_b_6h)
    kumo_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        # TK cross signals
        tk_cross_up = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_6h[i-1] <= kijun_6h[i-1])
        tk_cross_down = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_6h[i-1] >= kijun_6h[i-1])
        
        if position == 0:
            # Long: price above cloud + TK cross up + volume
            if (close[i] > kumo_top[i] and 
                tk_cross_up and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + volume
            elif (close[i] < kumo_bottom[i] and 
                  tk_cross_down and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below cloud bottom OR TK cross down
                if (close[i] < kumo_bottom[i] and close[i-1] >= kumo_bottom[i-1]) or \
                   tk_cross_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above cloud top OR TK cross up
                if (close[i] > kumo_top[i] and close[i-1] <= kumo_top[i-1]) or \
                   tk_cross_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals