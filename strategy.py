#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1wFilter
Hypothesis: Ichimoku TK cross with cloud filter on 6h, using 1w trend filter (price > 1w EMA50 for longs, < for shorts). 
Ichimoku provides dynamic support/resistance and momentum signals. Weekly trend filter ensures we only trade with the higher timeframe trend, reducing whipsaws in ranging markets. 
Works in bull markets (longs in uptrend) and bear markets (shorts in downtrend). 
Target: 12-30 trades/year (50-120 over 4 years) via strict TK cross + cloud + weekly trend confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align HTF indicators to 6h timeframe (completed 1w bar lag)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan, additional_delay_bars=1)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun, additional_delay_bars=1)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a, additional_delay_bars=1)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52 periods)
    start_idx = 53
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross (tenkan > kijun) AND price above cloud AND 1w uptrend (price > 1w EMA50)
            # Short: TK cross (tenkan < kijun) AND price below cloud AND 1w downtrend (price < 1w EMA50)
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
            price_above_cloud = close[i] > cloud_top
            price_below_cloud = close[i] < cloud_bottom
            trend_up = close[i] > ema50_1w_aligned[i]
            trend_down = close[i] < ema50_1w_aligned[i]
            
            long_signal = tk_cross_up and price_above_cloud and trend_up
            short_signal = tk_cross_down and price_below_cloud and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price closes below cloud (trend invalidation)
            exit_signal = close[i] < cloud_bottom
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price closes above cloud (trend invalidation)
            exit_signal = close[i] > cloud_top
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_1wFilter"
timeframe = "6h"
leverage = 1.0