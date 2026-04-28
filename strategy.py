#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1d
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud filter on 6h, using 1d Tenkan/Kijun/Senkou for trend filter. Works in bull/bear by avoiding trades against the higher timeframe cloud. Target: 20-40 trades/year per symbol.
"""

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku on 6h: Tenkan (9), Kijun (26), Senkou A/B (26, 52)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = (pd.Series(high).rolling(window=52, min_periods=52).max().values + 
                pd.Series(low).rolling(window=52, min_periods=52).min().values) / 2
    
    # Shift Senkou by 26 periods
    senkou_a = np.roll(senkou_a, 26)
    senkou_b = np.roll(senkou_b, 26)
    senkou_a[:26] = np.nan
    senkou_b[:26] = np.nan
    
    # Daily Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                   pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2
    
    senkou_a_1d = np.roll(senkou_a_1d, 26)
    senkou_b_1d = np.roll(senkou_b_1d, 26)
    senkou_a_1d[:26] = np.nan
    senkou_b_1d[:26] = np.nan
    
    # Align daily Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals on 6h
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Cloud: green (bullish) when senkou_a > senkou_b, red (bearish) when senkou_a < senkou_b
        cloud_green = senkou_a[i] > senkou_b[i]
        cloud_red = senkou_a[i] < senkou_b[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > senkou_a[i] and close[i] > senkou_b[i]
        price_below_cloud = close[i] < senkou_a[i] and close[i] < senkou_b[i]
        
        # Daily trend filter: price vs 1d cloud and TK cross
        price_above_1d_cloud = close[i] > senkou_a_1d_aligned[i] and close[i] > senkou_b_1d_aligned[i]
        price_below_1d_cloud = close[i] < senkou_a_1d_aligned[i] and close[i] < senkou_b_1d_aligned[i]
        tk_bullish_1d = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish_1d = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Long: bullish TK cross, price above cloud, and 1d bullish alignment
        long_signal = (tk_cross_up and price_above_cloud and 
                      (price_above_1d_cloud or tk_bullish_1d))
        
        # Short: bearish TK cross, price below cloud, and 1d bearish alignment
        short_signal = (tk_cross_down and price_below_cloud and 
                       (price_below_1d_cloud or tk_bearish_1d))
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1d"
timeframe = "6h"
leverage = 1.0