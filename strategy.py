#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend
6h strategy using Ichimoku cloud with Tenkan-Kijun cross and weekly trend filter.
- Long: TK cross above, price above cloud, weekly trend up (price > weekly EMA50)
- Short: TK cross below, price below cloud, weekly trend down (price < weekly EMA50)
- Exit: Opposite TK cross or price crosses cloud in opposite direction
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (trend following) and bear markets (counter-trend reversals)
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).mean().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).mean().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).mean().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).mean().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).mean().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).mean().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals to avoid look-ahead
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # TK cross conditions
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Cloud conditions
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up and price_above_cloud and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down and price_below_cloud and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down or price crosses below cloud
            if tk_cross_down or close[i] < cloud_bottom:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up or price crosses above cloud
            if tk_cross_up or close[i] > cloud_top:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0