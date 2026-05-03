#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d weekly pivot filter for BTC/ETH.
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h for trend and momentum.
# Weekly pivot levels (from 1d data) act as regime filter: only take longs above weekly pivot,
# shorts below weekly pivot. Reduces false signals in ranging markets.
# Designed for 50-150 total trades over 4 years with discrete position sizing.
# Focus on BTC/ETH as primary symbols; avoids SOL-only bias.

name = "6h_Ichimoku_1dWeeklyPivot_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for weekly pivot calculation (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior 1d OHLC (using last 5 completed 1d bars for weekly)
    # Weekly high = max of last 5 daily highs, weekly low = min of last 5 daily lows, weekly close = last daily close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window of 5 days for weekly aggregation
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h (wait for 1d bar to complete)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6s (already on 6h, no alignment needed)
    # But we need to shift Senkou Spans forward by 26 periods (they are plotted 26 periods ahead)
    # For trading, we use current Senkou A/B values (which were plotted 26 periods ago)
    # So we shift them BACK by 26 to align with current price
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a_lagged[i]
        senkou_b_val = senkou_b_lagged[i]
        wp_val = weekly_pivot_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(senkou_a_val) or np.isnan(senkou_b_val) or np.isnan(wp_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine Ichimoku trend: bullish if price above cloud, bearish if below cloud
        # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        is_bullish = close_val > cloud_top
        is_bearish = close_val < cloud_bottom
        
        # Weekly pivot filter: only take longs above weekly pivot, shorts below
        is_above_pivot = close_val > wp_val
        is_below_pivot = close_val < wp_val
        
        # Entry conditions: Tenkan-Kijun cross in direction of trend and pivot filter
        tenkan_cross_above = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
        tenkan_cross_below = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
        
        # Generate signals
        if position == 0:
            # Long: bullish Tenkan-Kijun cross, price above cloud, above weekly pivot
            if tenkan_cross_above and is_bullish and is_above_pivot:
                signals[i] = 0.25
                position = 1
            # Short: bearish Tenkan-Kijun cross, price below cloud, below weekly pivot
            elif tenkan_cross_below and is_bearish and is_below_pivot:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan-Kijun cross below OR price breaks below cloud bottom OR breaks below weekly pivot
            if tenkan_cross_below or close_val < cloud_bottom or close_val < wp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan-Kijun cross above OR price breaks above cloud top OR breaks above weekly pivot
            if tenkan_cross_above or close_val > cloud_top or close_val > wp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals