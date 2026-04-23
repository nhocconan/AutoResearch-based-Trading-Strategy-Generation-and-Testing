#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d TK Cross filter and volume confirmation.
Long when price breaks above Kumo (cloud) AND 1d TK Cross bullish AND volume > 1.5x 20-period MA.
Short when price breaks below Kumo AND 1d TK Cross bearish AND volume > 1.5x 20-period MA.
Exit when price re-enters the Kumo or TK Cross reverses.
Uses 1d HTF for TK Cross filter to ensure alignment with higher timeframe momentum.
Ichimoku provides dynamic support/resistance that adapts to volatility, working in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it's lagging
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Upper cloud: max(Senkou Span A, Senkou Span B)
    # Lower cloud: min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # Calculate 1d TK Cross (Tenkan/Kijun cross) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate 1d Tenkan-sen and Kijun-sen
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # 1d Tenkan-sen (9-period)
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen (26-period)
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d TK Cross: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish_1d = tenkan_sen_1d > kijun_sen_1d
    tk_bearish_1d = tenkan_sen_1d < kijun_sen_1d
    
    # Align 1d TK Cross to 6h timeframe
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish_1d.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish_1d.astype(float))
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Ichimoku 52-period, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 6h volume > 1.5x 20-period MA (moderate threshold)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper cloud AND 1d TK Cross bullish AND volume filter
            if close[i] > upper_cloud[i] and tk_bullish_aligned[i] > 0.5 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower cloud AND 1d TK Cross bearish AND volume filter
            elif close[i] < lower_cloud[i] and tk_bearish_aligned[i] > 0.5 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price re-enters cloud (below upper cloud) OR 1d TK Cross turns bearish
                if close[i] < upper_cloud[i] or tk_bullish_aligned[i] < 0.5:
                    exit_signal = True
            elif position == -1:
                # Short exit: price re-enters cloud (above lower cloud) OR 1d TK Cross turns bullish
                if close[i] > lower_cloud[i] or tk_bearish_aligned[i] < 0.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_KumoBreak_1dTKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0