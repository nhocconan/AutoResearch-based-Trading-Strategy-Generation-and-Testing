#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud + 12-hour ADX trend filter.
In trending markets (ADX > 25): trade in direction of TK cross when price is above/below cloud.
In ranging markets (ADX <= 25): fade price extremes at Kumo (cloud) boundaries.
Uses 6h for entry timing, 12h for trend strength and cloud calculation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_12h_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 12H TREND STRENGTH (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI values
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 12H ICHIMOKU CLOUD (HTF) ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (shift by 1 for completed bars)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    # Cloud boundaries
    upper_cloud = np.maximum(span_a_aligned, span_b_aligned)
    lower_cloud = np.minimum(span_a_aligned, span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need 52 periods for Senkou Span B
        if np.isnan(adx_aligned[i]) or np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]):
            signals[i] = 0.0
            continue
        
        # Determine market regime
        trending = adx_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: TK cross down OR price closes below cloud in trend OR price touches upper cloud in range
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
            price_below_cloud = close[i] < lower_cloud[i]
            price_at_upper_cloud = close[i] >= upper_cloud[i] * 0.995  # Near upper cloud
            
            if (trending and (tk_cross_down or price_below_cloud)) or \
               (not trending and price_at_upper_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross up OR price closes above cloud in trend OR price touches lower cloud in range
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
            price_above_cloud = close[i] > upper_cloud[i]
            price_at_lower_cloud = close[i] <= lower_cloud[i] * 1.005  # Near lower cloud
            
            if (trending and (tk_cross_up or price_above_cloud)) or \
               (not trending and price_at_lower_cloud):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # TK cross
            tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
            tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
            
            # Price position relative to cloud
            price_above_cloud = close[i] > upper_cloud[i]
            price_below_cloud = close[i] < lower_cloud[i]
            price_in_cloud = (close[i] >= lower_cloud[i]) and (close[i] <= upper_cloud[i])
            
            if trending:
                # In trending markets: trade TK cross in direction of trend
                if tk_cross_up and price_above_cloud:
                    position = 1
                    signals[i] = 0.25
                elif tk_cross_down and price_below_cloud:
                    position = -1
                    signals[i] = -0.25
            else:
                # In ranging markets: fade at cloud boundaries
                if price_at_upper_cloud := (close[i] >= upper_cloud[i] * 0.995):
                    position = -1
                    signals[i] = -0.25
                elif price_at_lower_cloud := (close[i] <= lower_cloud[i] * 1.005):
                    position = 1
                    signals[i] = 0.25
    
    return signals