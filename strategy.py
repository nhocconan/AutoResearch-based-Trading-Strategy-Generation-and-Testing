#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Weekly Pivot Direction Filter
# Uses 1d Ichimoku (Tenkan/Kijun/Senkou Span A/B) for trend regime and cloud filter.
# Enters long when price breaks above cloud in bull regime (price > weekly pivot),
# enters short when price breaks below cloud in bear regime (price < weekly pivot).
# Weekly pivot from 1w data provides structural bias to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years with discrete position sizing.

name = "6h_IchimokuCloud_1dWeeklyPivot_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku calculation (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h (wait for 1d bar to complete)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get 1w data for weekly pivot points (prior completed 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior completed 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C)/3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - low_1w
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot levels to 6h (wait for 1w bar to complete)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(senkou_a_val) or np.isnan(senkou_b_val) or
            np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine Ichimoku trend: bullish if Tenkan > Kijun
        is_ichimoku_bullish = tenkan_val > kijun_val
        # Determine cloud: green if Senkou A > Senkou B (bullish cloud)
        is_green_cloud = senkou_a_val > senkou_b_val
        # Determine price relative to cloud
        is_above_cloud = close_val > max(senkou_a_val, senkou_b_val)
        is_below_cloud = close_val < min(senkou_a_val, senkou_b_val)
        
        # Determine weekly pivot regime: bull if price > pivot, bear if price < pivot
        is_bull_regime = close_val > pivot_val
        is_bear_regime = close_val < pivot_val
        
        # Generate signals
        if position == 0:
            # Long entry: price breaks above cloud in bull regime with Ichimoku bullish bias
            if is_above_cloud and is_bull_regime and is_ichimoku_bullish:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below cloud in bear regime with Ichimoku bearish bias
            elif is_below_cloud and is_bear_regime and not is_ichimoku_bullish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below cloud or regime turns bearish
            if is_below_cloud or not is_bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above cloud or regime turns bullish
            if is_above_cloud or is_bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals