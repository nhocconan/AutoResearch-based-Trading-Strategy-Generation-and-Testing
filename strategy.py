#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Hypothesis: Trade 6h Ichimoku cloud breaks with 1d EMA50 trend filter and volume confirmation.
Ichimoku provides dynamic support/resistance via cloud (Senkou Span A/B) and momentum via TK cross.
1d EMA50 ensures trading with dominant daily trend to avoid counter-trend whipsaws.
Volume confirmation adds conviction to cloud breaks.
Works in bull (cloud breaks with trend) and bear (trend filter prevents false breaks in ranging markets).
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h (no additional shift needed as calculations are on 6h data)
    tenkan_sen_aligned = tenkan_sen
    kijun_sen_aligned = kijun_sen
    senkou_span_a_aligned = senkou_span_a
    senkou_span_b_aligned = senkou_span_b
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52), 1d EMA(50), volume MA(20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
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
        
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross
        tk_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        
        # Trend filter
        trend_up = close_val > ema_50_1d_aligned[i]
        trend_down = close_val < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud AND TK cross bullish AND volume confirm AND trend up
            long_signal = (close_val > cloud_top) and tk_cross and vol_conf and trend_up
            
            # Short: price breaks below cloud AND TK cross bearish AND volume confirm AND trend down
            short_signal = (close_val < cloud_bottom) and (not tk_cross) and vol_conf and trend_down
            
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
            # Exit: price breaks below cloud OR TK cross bearish OR trend flips down
            if (close_val < cloud_bottom) or (not tk_cross) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud OR TK cross bullish OR trend flips up
            if (close_val > cloud_top) or tk_cross or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0