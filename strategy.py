#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Ichimoku components, 1w for ADX trend strength.
- Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) calculated on 1d data.
- Long when price breaks above Kumo (cloud) AND Tenkan > Kijun (bullish TK cross) AND 1w ADX > 25.
- Short when price breaks below Kumo AND Tenkan < Kijun (bearish TK cross) AND 1w ADX > 25.
- In ranging markets (1w ADX < 20): fade at cloud edges with TK cross reversal.
- Volume confirmation: current volume > 1.3 * 20-period volume MA to avoid false breakouts.
- Discrete signal size: 0.25 to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2.0)
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 30, 20)  # Need enough bars for Ichimoku (52), ADX (30), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        
        # Kumo (cloud) boundaries
        upper_kumo = max(senkou_a, senkou_b)
        lower_kumo = min(senkou_a, senkou_b)
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above Kumo AND bullish TK cross
                    if curr_close > upper_kumo and tenkan > kijun:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below Kumo AND bearish TK cross
                    elif curr_close < lower_kumo and tenkan < kijun:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at cloud edges
                    # Long when price touches lower Kumo AND bullish TK cross (tenkan > kijun)
                    if curr_low <= lower_kumo and tenkan > kijun:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Kumo AND bearish TK cross (tenkan < kijun)
                    elif curr_high >= upper_kumo and tenkan < kijun:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Kumo OR TK cross turns bearish OR ADX drops to ranging
            if curr_close < lower_kumo or tenkan < kijun or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Kumo OR TK cross turns bullish OR ADX drops to ranging
            if curr_close > upper_kumo or tenkan > kijun or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_1wADXRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0