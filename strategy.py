#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Breakout_1dTrend_VolumeFilter
Hypothesis: Ichimoku cloud breakout with 1d trend filter (price > 1d EMA50) and volume confirmation. Works in bull/bear: cloud acts as dynamic S/R, trend filter ensures momentum alignment, volume prevents false breakouts. Target: 12-30 trades/year on 6h.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 30-period median (avoid low-volume breakouts)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    volume_filter = volume > vol_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (52 for Ichimoku, 50 for EMA, 30 for volume)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_median[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_filt = volume_filter[i]
        
        # Cloud boundaries: Senkou Span A and B
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price above/below cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema50_val
        downtrend = close_val < ema50_val
        
        # Entry conditions
        long_entry = tk_cross_up and price_above_cloud and uptrend and vol_filt
        short_entry = tk_cross_down and price_below_cloud and downtrend and vol_filt
        
        # Exit conditions: reverse TK cross or price re-enters cloud
        exit_long = tk_cross_down or not price_above_cloud
        exit_short = tk_cross_up or not price_below_cloud
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on signal
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short - exit on signal
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0