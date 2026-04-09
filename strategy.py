#!/usr/bin/env python3
# 6h_ichimoku_volume_confirmation_v2
# Hypothesis: Refined Ichimoku strategy with stricter entry conditions to reduce trade frequency.
# Uses daily Ichimoku Cloud for trend/regime filter, 6h TK cross for momentum, and volume confirmation.
# Added requirement: price must close beyond cloud by at least 0.5*ATR(6h) to avoid whipsaws.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 15-25 trades/year.
# Uses 1d HTF data for Ichimoku components, called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_volume_confirmation_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for daily Ichimoku (26*2)
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Daily Ichimoku parameters
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Align daily Ichimoku data to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h TK cross (Tenkan/Kijun cross) for entry timing
    tenkan_6h = (pd.Series(close).rolling(window=9, min_periods=9).max().values +
                 pd.Series(close).rolling(window=9, min_periods=9).min().values) / 2.0
    kijun_6h = (pd.Series(close).rolling(window=26, min_periods=26).max().values +
                pd.Series(close).rolling(window=26, min_periods=26).min().values) / 2.0
    tk_cross = tenkan_6h - kijun_6h  # Positive when Tenkan > Kijun (bullish)
    
    # 6h ATR for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr.rolling(window=14, min_periods=14).mean().values
    atr_6h[0] = np.nan  # First value invalid due to roll
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tk_cross[i]) or np.isnan(volume_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Cloud thickness as filter (avoid thin cloud whipsaws)
        cloud_thickness = cloud_top - cloud_bottom
        min_cloud_thickness = 0.5 * atr_6h[i]  # Require cloud thicker than half ATR
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR TK cross turns bearish
            if close[i] < cloud_bottom or tk_cross[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if close[i] > cloud_top or tk_cross[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and cloud_thickness > min_cloud_thickness:
                # Long entry: price above cloud by at least 0.5*ATR AND TK cross bullish
                if close[i] > cloud_top + 0.5 * atr_6h[i] and tk_cross[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below cloud by at least 0.5*ATR AND TK cross bearish
                elif close[i] < cloud_bottom - 0.5 * atr_6h[i] and tk_cross[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals