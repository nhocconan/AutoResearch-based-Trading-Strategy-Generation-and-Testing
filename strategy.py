#!/usr/bin/env python3
"""
6h_Ichimoku_CloudBreakout
Strategy: 6-hour Ichimoku breakout with 1d trend filter.
Long: Tenkan > Kijun + price above Kumo + price breaks above Kumo top
Short: Tenkan < Kijun + price below Kumo + price breaks below Kumo bottom
Kumo calculated from 1d data for trend context.
Position size: 0.25
Designed to capture momentum in trending markets while avoiding counter-trend trades.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Kumo breakout conditions
        kumo_breakout_up = close[i] > kumo_top[i] and close[i-1] <= kumo_top[i-1]
        kumo_breakout_down = close[i] < kumo_bottom[i] and close[i-1] >= kumo_bottom[i-1]
        
        if position == 0:
            # Long: bullish TK cross + price above Kumo + breaks above Kumo top + above 1d EMA50
            if (tenkan_above_kijun and price_above_kumo and 
                kumo_breakout_up and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below Kumo + breaks below Kumo bottom + below 1d EMA50
            elif (tenkan_below_kijun and price_below_kumo and 
                  kumo_breakout_down and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Kumo bottom or TK cross turns bearish
            if close[i] < kumo_bottom[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Kumo top or TK cross turns bullish
            if close[i] > kumo_top[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout"
timeframe = "6h"
leverage = 1.0