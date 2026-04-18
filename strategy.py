#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend
6h strategy using Ichimoku cloud and Tenkan/Kijun cross with weekly trend filter.
- Long: Tenkan crosses above Kijun, price above cloud, weekly close > weekly open
- Short: Tenkan crosses below Kijun, price below cloud, weekly close < weekly open
- Exit: Opposite cross or price crosses opposite cloud edge
Designed for ~10-25 trades/year per symbol (40-100 total over 4 years)
Works in bull markets (trend continuation) and bear markets (trend reversal)
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
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_uptrend = weekly_close > weekly_open
    weekly_downtrend = weekly_close < weekly_open
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen: (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B: (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span: current close shifted 26 periods back (not used in signals)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to avoid look-ahead
    # Tenkan and Kijun are based on current/past data, no shift needed
    # Senkou spans need alignment as they are plotted ahead
    tenkan_aligned = tenkan  # no look-ahead, uses past 9 periods
    kijun_aligned = kijun    # no look-ahead, uses past 26 periods
    cloud_top_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_a, additional_delay_bars=26)
    cloud_bottom_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_b, additional_delay_bars=26)
    
    # Cross signals
    tenkan_prev = np.roll(tenkan_aligned, 1)
    kijun_prev = np.roll(kijun_aligned, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_cross_up = (tenkan_aligned > kijun_aligned) & (tenkan_prev <= kijun_prev)
    tk_cross_down = (tenkan_aligned < kijun_aligned) & (tenkan_prev >= kijun_prev)
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top_aligned
    price_below_cloud = close < cloud_bottom_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough for Senkou B calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross up, price above cloud, weekly uptrend
            if tk_cross_up[i] and price_above_cloud[i] and weekly_uptrend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down, price below cloud, weekly downtrend
            elif tk_cross_down[i] and price_below_cloud[i] and weekly_downtrend_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross down or price drops below cloud bottom
            if tk_cross_down[i] or (close[i] < cloud_bottom_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross up or price rises above cloud top
            if tk_cross_up[i] or (close[i] > cloud_top_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0