#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Ichimoku Cloud and ADX trend.
- Ichimoku Cloud: Tenkan-sen (9-period) and Kijun-sen (26-period) from prior 1d candles.
- Breakout: Close > Senkou Span A (leading span 1) for long, Close < Senkou Span B (leading span 2) for short.
- Trend filter: Only trade breakouts in direction of 1d ADX(14) > 25 (strong trend).
- Volume confirmation: Current volume > 1.5x 20-period volume MA.
- Works in bull via buying cloud breakouts in uptrend, in bear via selling cloud breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components (using prior 1d OHLC)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Calculate ADX(14) for trend strength
    # True Range (TR)
    tr1 = pd.Series(df_1d['high']).values - pd.Series(df_1d['low']).values
    tr2 = abs(pd.Series(df_1d['high']).values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = abs(pd.Series(df_1d['low']).values - pd.Series(df_1d['close']).shift(1).values)
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1).values
    
    # Directional Movement (DM)
    dm_plus = pd.Series(df_1d['high']).values - pd.Series(df_1d['high']).shift(1).values
    dm_minus = pd.Series(df_1d['low']).shift(1).values - pd.Series(df_1d['low']).values
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed TR and DM
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_14 / tr14)
    di_minus = 100 * (dm_minus_14 / tr14)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h (Ichimoku lines are plotted current, ADX is lagging)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Ichimoku needs 52, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Ichimoku Cloud breakout with ADX trend filter and volume spike
            if volume_spike[i] and adx_aligned[i] > 25:
                # Long breakout: close > Senkou Span A (top of cloud) and Tenkan > Kijun (bullish)
                if close[i] > senkou_span_a_aligned[i] and tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < Senkou Span B (bottom of cloud) and Tenkan < Kijun (bearish)
                elif close[i] < senkou_span_b_aligned[i] and tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters cloud or opposite Ichimoku signal
            if close[i] < senkou_span_b_aligned[i]:  # Exit when price falls below cloud bottom
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters cloud or opposite Ichimoku signal
            if close[i] > senkou_span_a_aligned[i]:  # Exit when price rises above cloud top
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0