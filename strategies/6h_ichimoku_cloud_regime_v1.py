#!/usr/bin/env python3
# 6h_ichimoku_cloud_regime_v1
# Hypothesis: 6h strategy using 1d Ichimoku cloud with regime filter (ADX < 25 for range, ADX > 25 for trend).
# In ranging markets (ADX < 25): fade at cloud edges (short at Senkou Span A/B resistance, long at support).
# In trending markets (ADX > 25): breakout continuation (long when price breaks above cloud, short when breaks below).
# Uses volume confirmation (>1.3x 20-period average) to reduce false signals.
# Designed for low turnover (target: 50-150 total trades over 4 years) by requiring regime alignment and volume spike.
# Works in both bull and bear markets via regime adaptation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

name = "6h_ichimoku_cloud_regime_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate daily Ichimoku components
    h1 = df_1d['high'].values
    l1 = df_1d['low'].values
    c1 = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(h1, l1, c1)
    
    # Align HTF Ichimoku to LTF (6h)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # 6h LTF ADX for regime detection (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.abs(high[0] - low[0])
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(tenkan_1d_aligned[i]) or
            np.isnan(kijun_1d_aligned[i]) or np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price breaks below cloud bottom OR adverse regime shift
            if close[i] < cloud_bottom or (adx[i] < 25 and close[i] < tenkan_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above cloud top OR adverse regime shift
            if close[i] > cloud_top or (adx[i] < 25 and close[i] > tenkan_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Regime filter: ADX > 25 = trending, ADX < 25 = ranging
                if adx[i] > 25:  # Trending regime
                    # Enter long on breakout above cloud
                    if close[i] > cloud_top:
                        position = 1
                        signals[i] = 0.25
                    # Enter short on breakdown below cloud
                    elif close[i] < cloud_bottom:
                        position = -1
                        signals[i] = -0.25
                else:  # Ranging regime (ADX < 25)
                    # Fade at cloud edges: short at resistance, long at support
                    if close[i] > cloud_top and close[i] < senkou_a_1d_aligned[i] * 1.02:
                        position = -1
                        signals[i] = -0.25
                    elif close[i] < cloud_bottom and close[i] > senkou_b_1d_aligned[i] * 0.98:
                        position = 1
                        signals[i] = 0.25
    
    return signals