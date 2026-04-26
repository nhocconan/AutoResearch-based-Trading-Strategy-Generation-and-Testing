#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_v1
Hypothesis: 6h Ichimoku cloud breakout with weekly trend filter for BTC/ETH.
- Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND weekly trend is up
- Short when price breaks below Ichimoku cloud AND weekly trend is down
- Uses Ichimoku calculated on 6h chart for entry signals
- Weekly trend filter (price vs weekly EMA20) avoids counter-trend trades in bear markets
- Designed for lower frequency (target 12-37 trades/year on 6h) to minimize fee drag
- Novelty: Ichimoku cloud breakout on 6h with weekly trend filter avoids whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter (needs completed weekly candle)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    # Weekly trend: 1 = uptrend (close > EMA20), -1 = downtrend (close < EMA20), 0 = invalid
    trend_1w = np.where(ema_20_1w_aligned > 0, 
                        np.where(close > ema_20_1w_aligned, 1, -1), 
                        0)
    
    # Calculate Ichimoku components on 6h chart (primary timeframe)
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
    
    # The actual cloud boundaries for current period (shifted back 26 periods)
    # We need to align the Senkou spans properly - they are plotted 26 periods ahead
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for shift)
    start_idx = 52 + 26  # 78 bars to ensure all Ichimoku components are valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku cloud breakout conditions with weekly trend filter
        if position == 0:
            # Long: Price breaks above cloud AND weekly uptrend
            if close[i] > cloud_top[i] and trend_1w[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND weekly downtrend
            elif close[i] < cloud_bottom[i] and trend_1w[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below cloud bottom OR weekly trend turns down
            if close[i] < cloud_bottom[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above cloud top OR weekly trend turns up
            if close[i] > cloud_top[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_v1"
timeframe = "6h"
leverage = 1.0