# 6h_20241216_Ichimoku_CloudBreak_1dTrend
# Hypothesis: Ichimoku cloud breakout on 6h with daily trend filter. Combines Tenkan/Kijun cross with cloud color filter for trend confirmation. Uses weekly timeframe for higher-order trend filter to avoid counter-trend trades. Designed for 60-120 total trades over 4 years (15-30/year) with focus on clean breaks in both bull and bear markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_20241216_Ichimoku_CloudBreak_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data once for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for higher-order trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Weekly trend filter: EMA(25) on weekly close
    close_1w = df_1w['close'].values
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(ema25_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        ema25_1w_val = ema25_1w_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Enter long: Tenkan crosses above Kijun AND price above cloud AND weekly uptrend
            if (tenkan_val > kijun_val and 
                close[i] > cloud_top and 
                close[i] > ema25_1w_val):
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan crosses below Kijun AND price below cloud AND weekly downtrend
            elif (tenkan_val < kijun_val and 
                  close[i] < cloud_bottom and 
                  close[i] < ema25_1w_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price drops below cloud
            if (tenkan_val < kijun_val or close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan_val > kijun_val or close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals