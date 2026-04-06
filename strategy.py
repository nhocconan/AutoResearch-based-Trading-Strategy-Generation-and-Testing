#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku cloud with 1-week trend filter and volume confirmation.
# Uses 1-week Ichimoku cloud (from daily data) for primary trend bias.
# 6-hour Tenkan/Kijun cross for entry timing with volume confirmation.
# Designed to work in both bull and bear markets via cloud filter.
# Targets 50-150 total trades over 4 years with strict entry conditions.

name = "6h_ichimoku1w_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week Ichimoku components (from daily data)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Tenkan-sen (9-period) and Kijun-sen (26-period) on weekly
    def calculate_ichimoku(high_arr, low_arr, close_arr):
        n1 = len(high_arr)
        tenkan = np.full(n1, np.nan)
        kijun = np.full(n1, np.nan)
        senkou_a = np.full(n1, np.nan)
        senkou_b = np.full(n1, np.nan)
        
        # Tenkan-sen: (9-period high + 9-period low)/2
        for i in range(8, n1):
            tenkan[i] = (np.max(high_arr[i-8:i+1]) + np.min(low_arr[i-8:i+1])) / 2
        
        # Kijun-sen: (26-period high + 26-period low)/2
        for i in range(25, n1):
            kijun[i] = (np.max(high_arr[i-25:i+1]) + np.min(low_arr[i-25:i+1])) / 2
        
        # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
        for i in range(n1):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                idx = i + 26
                if idx < n1:
                    senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
        
        # Senkou Span B: (52-period high + 52-period low)/2 shifted 26 periods ahead
        for i in range(51, n1):
            senkou_b[i] = (np.max(high_arr[i-51:i+1]) + np.min(low_arr[i-51:i+1])) / 2
        
        for i in range(n1):
            if not np.isnan(senkou_b[i]):
                idx = i + 26
                if idx < n1:
                    senkou_b[idx] = senkou_b[i]
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # 6-hour Tenkan-sen and Kijun-sen for entry signals
    tenkan_6h = np.full(n, np.nan)
    kijun_6h = np.full(n, np.nan)
    
    for i in range(8, n):
        tenkan_6h[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    for i in range(25, n):
        kijun_6h[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Start after Ichimoku calculations are valid
        # Skip if required data not available
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        bullish_cloud = senkou_a_1w_aligned[i] > senkou_b_1w_aligned[i]
        bearish_cloud = senkou_a_1w_aligned[i] < senkou_b_1w_aligned[i]
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below cloud or Tenkan-Kijun cross down
            if (close[i] < senkou_a_1w_aligned[i] and close[i] < senkou_b_1w_aligned[i]) or \
               (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above cloud or Tenkan-Kijun cross up
            if (close[i] > senkou_a_1w_aligned[i] and close[i] > senkou_b_1w_aligned[i]) or \
               (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and cloud filter
            if volume_filter:
                # Bullish TK cross in bullish cloud
                if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                    bullish_cloud and close[i] > senkou_a_1w_aligned[i] and close[i] > senkou_b_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Bearish TK cross in bearish cloud
                elif (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                      bearish_cloud and close[i] < senkou_a_1w_aligned[i] and close[i] < senkou_b_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals