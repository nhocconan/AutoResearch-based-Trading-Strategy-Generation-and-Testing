#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrendFilter_v2
Hypothesis: 6h Ichimoku Kumo twist strategy with 1d trend filter.
- Uses Kumo twist (Senkou Span A/B cross) as early trend change signal
- Filters by 1d EMA50 trend to avoid counter-trend trades
- Enters when price confirms the twist by breaking the Kijun-sen
- Exits on opposite Kumo twist or when price re-enters the cloud
- Designed for low turnover (target: 50-150 trades over 4 years) to minimize fee drag
- Works in bull/bear markets by aligning with higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku calculations (9, 26, 52 periods)
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
    
    # Kumo twist detection: Senkou A/B cross
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_prev >= senkou_b_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(bullish_twist[i]) or np.isnan(bearish_twist[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Price position relative to Kijun
        price_above_kijun = close[i] > kijun[i]
        price_below_kijun = close[i] < kijun[i]
        
        # 1d trend filter
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: bullish Kumo twist AND price above Kijun AND 1d uptrend
            if bullish_twist[i] and price_above_kijun and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo twist AND price below Kijun AND 1d downtrend
            elif bearish_twist[i] and price_below_kijun and downtrend_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bearish Kumo twist OR price falls below Kijun
            if bearish_twist[i] or price_below_kijun:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish Kumo twist OR price rises above Kijun
            if bullish_twist[i] or price_above_kijun:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrendFilter_v2"
timeframe = "6h"
leverage = 1.0