#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_v1
Hypothesis: 6h Ichimoku TK cross with 1d cloud twist filter for trend confirmation.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when TK crosses above AND price is above cloud AND 1d trend is up (close > EMA50)
- Short when TK crosses below AND price is below cloud AND 1d trend is down (close < EMA50)
- Ichimoku cloud (Senkou Span A/B) acts as dynamic support/resistance
- 1d EMA50 trend filter reduces whipsaw and aligns with higher timeframe momentum
- Designed for low trade frequency with proven edge on BTC/ETH from Ichimoku's trend-following nature
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
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
    
    # Cloud (Kumo) boundaries
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_cross_below = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions with 1d trend filter
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        
        # 1d trend filter
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: TK cross above AND price above cloud AND 1d uptrend
            if tk_cross_above[i] and price_above_cloud and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND price below cloud AND 1d downtrend
            elif tk_cross_below[i] and price_below_cloud and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross below OR price falls below cloud OR 1d trend turns down
            if tk_cross_below[i] or not price_above_cloud or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross above OR price rises above cloud OR 1d trend turns up
            if tk_cross_above[i] or not price_below_cloud or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_v1"
timeframe = "6h"
leverage = 1.0