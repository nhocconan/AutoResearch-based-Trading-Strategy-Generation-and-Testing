#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, Ichimoku cloud breakouts aligned with weekly trend (price above/below weekly Kumo) 
with volume confirmation capture strong momentum moves. Weekly trend filter reduces false breakouts in ranging markets, 
while volume spike ensures institutional participation. Discrete position sizing (0.25) limits trades to ~12-30/year 
to minimize fee drag. Ichimoku provides dynamic support/resistance via Kumo cloud, effective in both bull and bear 
markets by adapting to volatility.
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
    volume = prices['volume'].values
    
    # Weekly data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Ichimoku components for trend filter
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tenkan_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1w).rolling(window=9, min_periods=9).min()).values / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1w).rolling(window=26, min_periods=26).min()).values / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2)
    # Weekly trend: price above/both Senkou spans = bullish, below both = bearish
    weekly_bullish = (close > senkou_a_1w) & (close > senkou_b_1w)
    weekly_bearish = (close < senkou_a_1w) & (close < senkou_b_1w)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 6h Ichimoku components for entry signals
    # Tenkan-sen (6h)
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()).values / 2
    # Kijun-sen (6h)
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()).values / 2
    # Senkou Span A (6h)
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2)
    # Senkou Span B (6h)
    senkou_b_6h = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    # Kumo cloud boundaries (future shift handled by alignment in Ichimoku definition)
    # For breakout: price breaks above/below current cloud
    cloud_top_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # 6h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Ichimoku (52) + volume MA (20) + aligned HTF arrays
    start_idx = max(52, 20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top_6h[i]) or np.isnan(cloud_bottom_6h[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above Kumo cloud with volume spike and weekly bullish trend
            long_breakout = (curr_close > cloud_top_6h[i]) and vol_spike[i] and weekly_bullish_aligned[i] > 0.5
            # Short: price breaks below Kumo cloud with volume spike and weekly bearish trend
            short_breakout = (curr_close < cloud_bottom_6h[i]) and vol_spike[i] and weekly_bearish_aligned[i] > 0.5
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Kumo cloud OR weekly trend turns bearish
            if (curr_close < cloud_bottom_6h[i]) or (weekly_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Kumo cloud OR weekly trend turns bullish
            if (curr_close > cloud_top_6h[i]) or (weekly_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0