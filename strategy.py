#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with Daily Trend Filter
# Takes long when price is above Kumo (cloud) and Tenkan > Kijun with daily price above weekly EMA50
# Takes short when price is below Kumo and Tenkan < Kijun with daily price below weekly EMA50
# Exits when price crosses back into the cloud or Tenkan/Kijun cross reverses
# Ichimoku provides dynamic support/resistance; daily trend filter ensures alignment with higher timeframe
# Works in bull markets via trend following, in bear via short signals from cloud breakdowns
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for entry)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Align daily and weekly EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 periods for Senkou B)
    start = 52
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Check if price is above/below cloud
        above_cloud = price > upper_cloud
        below_cloud = price < lower_cloud
        
        # Check Tenkan/Kijun cross
        tenkan_gt_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_lt_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Check daily and weekly trend (price above/below EMA50)
        # Note: For weekly, we need to handle the case where i might be out of bounds for the aligned array
        # But align_htf_to_ltf should handle this by returning NaN for out-of-bounds
        daily_uptrend = close[i] > ema50_1d_aligned[i] if not np.isnan(ema50_1d_aligned[i]) else False
        weekly_uptrend = close[i] > ema50_1w_aligned[i] if not np.isnan(ema50_1w_aligned[i]) else False
        
        if position == 0:
            # Long setup: price above cloud, Tenkan > Kijun, and daily/weekly uptrend
            if above_cloud and tenkan_gt_kijun and daily_uptrend and weekly_uptrend:
                position = 1
                signals[i] = position_size
            # Short setup: price below cloud, Tenkan < Kijun, and daily/weekly downtrend
            elif below_cloud and tenkan_lt_kijun and not daily_uptrend and not weekly_uptrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below cloud or Tenkan < Kijun
            if not above_cloud or tenkan_lt_kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above cloud or Tenkan > Kijun
            if not below_cloud or tenkan_gt_kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Ichimoku_Cloud_DailyTrend"
timeframe = "6h"
leverage = 1.0