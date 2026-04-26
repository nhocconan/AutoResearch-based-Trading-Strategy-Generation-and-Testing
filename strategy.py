#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_v1
Hypothesis: Ichimoku cloud breakout on 6h with weekly trend filter and volume confirmation. Works in both bull (price above cloud with TK cross up) and bear (price below cloud with TK cross down) by using the cloud as dynamic support/resistance and TK cross for momentum. Weekly trend filter avoids counter-trend trades. Volume confirmation ensures institutional participation. Targets 12-30 trades/year on 6h timeframe to minimize fee drag.
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
    
    # Get weekly data for trend filter (more stable than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period_9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period_9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period_9_high + period_9_low) / 2.0
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period_26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period_26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period_26_high + period_26_low) / 2.0
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period_52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period_52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period_52_high + period_52_low) / 2.0
    
    # Lagging Span (Chikou Span): close plotted 26 periods behind
    # Not used for signals as it's lagging
    
    # Align Ichimoku components to current bars (no look-ahead)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b)
    
    # Weekly trend filter: price above/below weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku calculations (52) and weekly EMA
    start_idx = max(52, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or
            np.isnan(span_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
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
        # Cloud top and bottom
        cloud_top = max(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = min(span_a_aligned[i], span_b_aligned[i])
        
        # TK cross: Tenkan crossing above/below Kijun
        tk_cross_up = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_cross_down = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Price relative to cloud
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # Weekly trend
        weekly_uptrend = close_val > ema_50_1w_aligned[i]
        weekly_downtrend = close_val < ema_50_1w_aligned[i]
        
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above cloud, TK cross up, weekly uptrend, volume spike
            long_signal = price_above_cloud and tk_cross_up and weekly_uptrend and vol_spike
            
            # Short: price below cloud, TK cross down, weekly downtrend, volume spike
            short_signal = price_below_cloud and tk_cross_down and weekly_downtrend and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR TK cross down OR weekly trend flips down
            if (not price_above_cloud) or (not tk_cross_up) or (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK cross up OR weekly trend flips up
            if (not price_below_cloud) or (not tk_cross_down) or (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0