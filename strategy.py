#!/usr/bin/env python3
name = "6h_Ichimoku_Tenkan_Kijun_Cloud_Filter_1d"
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
    
    # Calculate 1d Ichimoku components for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
              pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
             pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    senkou_b = senkou_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 6h EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema20[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        green_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]  # Bullish cloud
        red_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]    # Bearish cloud
        above_cloud = close[i] > max(senkou_a_aligned[i], senkou_b_aligned[i])
        below_cloud = close[i] < min(senkou_a_aligned[i], senkou_b_aligned[i])
        in_cloud = (close[i] >= min(senkou_a_aligned[i], senkou_b_aligned[i]) and 
                    close[i] <= max(senkou_a_aligned[i], senkou_b_aligned[i]))
        
        # Tenkan/Kijun cross
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: TK cross up + price above cloud + bullish cloud
            if tk_cross_up and above_cloud and green_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + bearish cloud
            elif tk_cross_down and below_cloud and red_cloud:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: TK cross down OR price below cloud
                if tk_cross_down or below_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: TK cross up OR price above cloud
                if tk_cross_up or above_cloud:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals