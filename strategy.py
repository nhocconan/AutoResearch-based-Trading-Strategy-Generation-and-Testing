#!/usr/bin/env python3
"""
1d_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter
Hypothesis: Ichimoku cloud breakouts on daily timeframe with weekly trend filter capture sustained moves while avoiding counter-trend trades. 
Price above/below cloud with TK cross confirms momentum. Weekly trend (price vs weekly Kijun) ensures trend alignment. 
Weekly timeframe reduces noise, daily provides timely signals. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal generation, we use current price vs cloud
    
    # Weekly trend filter: price vs weekly Kijun (26-period)
    period26_high_w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_w = (period26_high_w + period26_low_w) / 2
    
    # Align Ichimoku components to daily timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    kijun_w_aligned = align_htf_to_ltf(prices, df_1w, kijun_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 52 periods for Senkou B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(kijun_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        kijun_w_val = kijun_w_aligned[i]
        
        # Cloud boundaries: Senkou A and Senkou B
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # TK Cross: Tenkan crossing Kijun
        tk_cross_up = tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price vs cloud
        price_above_cloud = close_val > upper_cloud
        price_below_cloud = close_val < lower_cloud
        
        # Weekly trend filter
        weekly_uptrend = close_val > kijun_w_val
        weekly_downtrend = close_val < kijun_w_val
        
        if position == 0:
            # Long: price above cloud + TK cross up + weekly uptrend
            if price_above_cloud and tk_cross_up and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: price below cloud + TK cross down + weekly downtrend
            elif price_below_cloud and tk_cross_down and weekly_downtrend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: price drops below cloud OR TK cross down
            if not price_above_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price rises above cloud OR TK cross up
            if not price_below_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0