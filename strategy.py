#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v1
Hypothesis: Trade 6h Ichimoku cloud breaks in direction of 1d trend (EMA50) with volume confirmation.
Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross). 
In bull markets: break above cloud + TK cross bullish + 1d uptrend = long.
In bear markets: break below cloud + TK cross bearish + 1d downtrend = short.
Volume filter ensures breakout conviction. Targets 50-150 total trades over 4 years.
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # For signals, we use current close vs cloud
    
    # The cloud is between Senkou A and Senkou B
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    # But Senkou A/B are shifted 26 periods ahead, so to get current cloud values:
    # We need values that were calculated 26 periods ago
    upper_cloud = np.roll(np.maximum(senkou_a, senkou_b), 26)
    lower_cloud = np.roll(np.minimum(senkou_a, senkou_b), 26)
    # First 26 values will be rolled from end - set to NaN
    upper_cloud[:26] = np.nan
    lower_cloud[:26] = np.nan
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52), 1d EMA(50), volume MA(20)
    start_idx = max(52, 50, 20) + 26  # +26 for cloud shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan[i]) or
            np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or
            np.isnan(lower_cloud[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        tk_bull = tk_cross_above[i]
        tk_bear = tk_cross_below[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        
        # Cloud breakout conditions
        above_cloud = close_val > upper_cloud[i]
        below_cloud = close_val < lower_cloud[i]
        
        if position == 0:
            # Long: price breaks above cloud AND TK bullish cross AND 1d uptrend AND volume
            long_signal = above_cloud and tk_bull and trend_up and vol_conf
            
            # Short: price breaks below cloud AND TK bearish cross AND 1d downtrend AND volume
            short_signal = below_cloud and tk_bear and trend_down and vol_conf
            
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
            # Exit: price falls below cloud OR TK bearish cross OR 1d trend flips down
            if (close_val < lower_cloud[i]) or tk_bear or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK bullish cross OR 1d trend flips up
            if (close_val > upper_cloud[i]) or tk_bull or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0