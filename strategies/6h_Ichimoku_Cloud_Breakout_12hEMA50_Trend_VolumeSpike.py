#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance on 12h timeframe.
Price breaking above/below cloud with TK cross confirmation, volume spike,
and aligned with 12h EMA50 trend captures momentum in both bull and bear markets.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Designed for 6h timeframe with tight entry conditions to achieve 12-37 trades/year.
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
    
    # Get 12h data for Ichimoku and EMA (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_12h['high'].values).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_12h['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_12h['high'].values).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_12h['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_12h['high'].values).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_12h['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    # Calculate EMA50 on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52 periods) and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if position == 0:
            # Look for entry signals
            # TK Cross: Tenkan crosses above/below Kijun
            tk_cross_up = tenkan > kijun
            tk_cross_down = tenkan < kijun
            
            # Long: price breaks above cloud AND TK cross up AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_close > cloud_top) and tk_cross_up and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below cloud AND TK cross down AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_close < cloud_bottom) and tk_cross_down and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below cloud OR TK cross down OR price crosses below EMA
            if (curr_close < cloud_bottom) or (tenkan < kijun) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above cloud OR TK cross up OR price crosses above EMA
            if (curr_close > cloud_top) or (tenkan > kijun) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0