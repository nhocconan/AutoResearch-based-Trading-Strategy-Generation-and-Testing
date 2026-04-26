#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v1
Hypothesis: Trade 6h Ichimoku cloud breaks with 1d EMA50 trend filter for BTC/ETH.
In bull markets: price above cloud + bullish TK cross + 1d uptrend = long.
In bear markets: price below cloud + bearish TK cross + 1d downtrend = short.
Ichimoku cloud acts as dynamic support/resistance; TK cross signals momentum.
1d EMA50 ensures trading with higher timeframe trend to avoid counter-trend whipsaws.
Targets 80-160 total trades over 4 years (20-40/year) with signal size 0.25.
Uses 6h timeframe to balance trade frequency and capture multi-day moves.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods) on 6h
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
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for entry, but could be used for confirmation
    
    # Cloud top/bottom: Senkou Span A/B shifted forward 26 periods
    # For entry at bar i, we use cloud values from i-26 (already published)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid (rolled from end)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Cloud top is max of Senkou A/B, bottom is min
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) and 1d EMA50
    start_idx = max(52, 50) + 26  # +26 for cloud shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan[i]) or
            np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross AND 1d uptrend
            long_signal = (close_val > cloud_top[i]) and tk_cross_up[i] and trend_1d_up
            
            # Short: price below cloud AND bearish TK cross AND 1d downtrend
            short_signal = (close_val < cloud_bottom[i]) and tk_cross_down[i] and trend_1d_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR 1d trend flips down
            if close_val < cloud_bottom[i] or not trend_1d_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR 1d trend flips up
            if close_val > cloud_top[i] or not trend_1d_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0