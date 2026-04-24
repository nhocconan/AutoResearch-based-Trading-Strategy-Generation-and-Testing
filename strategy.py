#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Ichimoku cloud and trend filter.
- Ichimoku components calculated on 1d: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52).
- Long when price breaks above Senkou Span A (top of cloud) with volume spike, 
  Short when price breaks below Senkou Span B (bottom of cloud) with volume spike.
- Trend filter: Only trade in direction of 1d Kijun-sen slope (long if rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Ichimoku cloud acts as dynamic support/resistance that adapts to volatility,
  working in both bull (buying cloud breakouts in uptrend) and bear (selling cloud breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h (each 1d bar = 4x 6h bars)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Ichimoku + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d Kijun-sen trend
            if i > 0 and not np.isnan(kijun_sen_aligned[i-1]):
                kijun_slope = kijun_sen_aligned[i] - kijun_sen_aligned[i-1]
                if kijun_slope > 0:  # Uptrend
                    # Long when price breaks above Senkou Span A (top of cloud) with volume spike
                    if close[i] > senkou_span_a_aligned[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif kijun_slope < 0:  # Downtrend
                    # Short when price breaks below Senkou Span B (bottom of cloud) with volume spike
                    if close[i] < senkou_span_b_aligned[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below Senkou Span B (bottom of cloud)
            if close[i] < senkou_span_b_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Senkou Span A (top of cloud)
            if close[i] > senkou_span_a_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dKijun_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0