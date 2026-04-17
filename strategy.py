#!/usr/bin/env python3
"""
4h_Ichimoku_Cloud_Breakout_TF
Ichimoku Cloud breakout with trend confirmation from daily Ichimoku and volume spike.
Trades on breakouts above/below cloud with TK cross confirmation and volume filter.
Designed for 4h timeframe with daily trend filter to reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 4h Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used in signals)
    
    # === Daily Ichimoku for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Tenkan-sen (9-period)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    # Daily Kijun-sen (26-period)
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Daily Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Daily Senkou Span B (52-period)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma_20 * 1.8  # volume spike: 1.8x average
    
    # === Align daily Ichimoku components to 4h ===
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Daily cloud top and bottom for trend filter
        daily_cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        daily_cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above cloud, TK cross up, above daily cloud, volume confirmation
            if (close[i] > cloud_top and 
                tk_cross_up and 
                close[i] > daily_cloud_top and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below cloud, TK cross down, below daily cloud, volume confirmation
            elif (close[i] < cloud_bottom and 
                  tk_cross_down and 
                  close[i] < daily_cloud_bottom and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price falls below cloud OR TK cross down
            if (close[i] < cloud_bottom or 
                tk_cross_down):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud OR TK cross up
            if (close[i] > cloud_top or 
                tk_cross_up):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Ichimoku_Cloud_Breakout_TF"
timeframe = "4h"
leverage = 1.0