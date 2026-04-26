#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v1
Hypothesis: On 6h timeframe, trade Ichimoku cloud breakouts with 1d EMA50 trend filter. 
Long when price breaks above cloud in 1d uptrend, short when breaks below cloud in 1d downtrend. 
Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross) while 1d trend filter 
avoids counter-trend trades. Designed for low frequency (12-30 trades/year) to minimize fee drag 
and work in both bull/bear markets via trend alignment.
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
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): not needed for breakout signals
    
    # Cloud (Kumo) boundaries: Senkou Span A and B shifted forward 26 periods
    # For breakout detection, we need current cloud (Senkou A/B shifted back 26 periods)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52 for Senkou B) + 26 shift + 1d EMA50
    start_idx = max(52, 26) + 26 + 50  # 154
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or
            np.isnan(cloud_bottom_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        cloud_top_val = cloud_top_aligned[i]
        cloud_bottom_val = cloud_bottom_aligned[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Cloud breakout signals
        bullish_breakout = close_val > cloud_top_val
        bearish_breakout = close_val < cloud_bottom_val
        
        if position == 0:
            # Long: price breaks above cloud in 1d uptrend
            if bullish_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud in 1d downtrend
            elif bearish_breakout and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit when price breaks below cloud OR trend changes
            signals[i] = 0.25
            if close_val < cloud_bottom_val or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit when price breaks above cloud OR trend changes
            signals[i] = -0.25
            if close_val > cloud_top_val or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0