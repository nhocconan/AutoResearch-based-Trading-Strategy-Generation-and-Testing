#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, use Ichimoku cloud from 1d for structural support/resistance, with 1w trend filter (ADX > 25) and volume confirmation (>2.0x 20-period average). Enter long when price breaks above the cloud (Senkou Span A) with bullish TK cross, 1w uptrend, and volume spike. Enter short when price breaks below the cloud (Senkou Span B) with bearish TK cross, 1w downtrend, and volume spike. Uses discrete position size 0.25 to balance capture and drawdown. Designed for 12-30 trades/year on 6h by requiring weekly alignment and volume confirmation, reducing overtrading while capturing structured moves in both bull and bear markets.
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
    
    # Get 1d data for Ichimoku cloud and 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 52 or len(df_1w) < 50:  # Need enough for Ichimoku and ADX
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 1w ADX for trend filter (ADX > 25 indicates strong trend)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = high_1w - high_1w.shift(1)
    dm_minus = low_1w.shift(1) - low_1w
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    tr_smoothed = tr.rolling(window=14, min_periods=14).mean()
    dm_plus_smoothed = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smoothed = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smoothed / tr_smoothed)
    di_minus = 100 * (dm_minus_smoothed / tr_smoothed)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52), ADX warmup (14+14), volume MA warmup (20)
    start_idx = max(52, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # 1w trend alignment
        trend_strong = adx_aligned[i] > 25
        trend_1w_uptrend = trend_strong and (di_plus.values[i] > di_minus.values[i]) if not (np.isnan(di_plus.values[i]) or np.isnan(di_minus.values[i])) else False
        trend_1w_downtrend = trend_strong and (di_minus.values[i] > di_plus.values[i]) if not (np.isnan(di_plus.values[i]) or np.isnan(di_minus.values[i])) else False
        
        if position == 0:
            # Long: price breaks above cloud + bullish TK cross + 1w uptrend + volume spike
            long_signal = price_above_cloud and tk_bullish and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below cloud + bearish TK cross + 1w downtrend + volume spike
            short_signal = price_below_cloud and tk_bearish and trend_1w_downtrend and volume_spike[i]
            
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
            # Exit: price breaks below cloud OR TK cross turns bearish OR trend weakens
            if (price_below_cloud or not tk_bullish or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud OR TK cross turns bullish OR trend weakens
            if (price_above_cloud or not tk_bearish or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0