#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with daily filter and volume confirmation
# Uses Ichimoku conversion (9), base (26), lagging span (52) for trend signals
# Confirms with 1d cloud (Senkou A/B) direction to filter trend alignment
# Requires volume > 1.5x 20-period average for institutional participation
# Targets 80-140 total trades over 4 years (20-35/year) to avoid fee drag
# Works in bull/bear via cloud filter + momentum confirmation

name = "6h_Ichimoku_1dCloud_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Ichimoku components (6h)
    # Conversion Line (tenkan): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Base Line (kijun): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou A): (tenkan + kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Leading Span B (Senkou B): (52-period high + low)/2 shifted 26 periods
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d Ichimoku cloud components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Conversion and Base
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # 1d Senkou A and B (current cloud)
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    # Cloud top/bottom (current)
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    # Align 1d cloud to 6h
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish: price above cloud AND tenkan > kijun (bullish momentum)
            bullish = (close[i] > cloud_top_aligned[i]) and (tenkan_aligned[i] > kijun_aligned[i])
            # Bearish: price below cloud AND tenkan < kijun (bearish momentum)
            bearish = (close[i] < cloud_bottom_aligned[i]) and (tenkan_aligned[i] < kijun_aligned[i])
            
            if bullish and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif bearish and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below cloud OR tenkan < kijun (momentum loss)
            if (close[i] < cloud_top_aligned[i]) or (tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above cloud OR tenkan > kijun (momentum loss)
            if (close[i] > cloud_bottom_aligned[i]) or (tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals