#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy using 1w Ichimoku cloud (from daily data) for trend filter and 6h price action for entry.
# Uses Tenkan/Kijun cross and price relative to Kumo (cloud) from 1d timeframe to avoid counter-trend trades.
# Enters only on 6h timeframe during 08-20 UTC session with volume confirmation.
# Designed to work in both bull (trend following) and bear (avoiding false signals via cloud filter).
# Target: 50-150 total trades over 4 years (~12-37/year) with position size 0.25.
name = "6h_1d_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku components (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters: Tenkan=9, Kijun=26, Senkou B=52
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6s timeframe (wait for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # For cloud twist handling, we use the actual values without shifting in alignment
    # The alignment function already handles the look-ahead prevention
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_kumo = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan > Kijun AND price above Kumo (bullish)
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > upper_kumo and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price below Kumo (bearish)
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < lower_kumo and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Tenkan < Kijun OR price drops below Kumo
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < lower_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Tenkan > Kijun OR price rises above Kumo
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > upper_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals