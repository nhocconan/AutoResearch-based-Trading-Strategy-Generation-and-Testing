#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend and 1d for Camarilla levels.
- Camarilla levels from 1d: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2.
  Breakout above H3 with volume confirmation = long, breakdown below L3 = short.
- Trend filter: Only trade in direction of 1w EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 50-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying H3 breakouts in uptrend, in bear via selling L3 breakdowns in downtrend.
- Uses weekly EMA for smooth trend filter to avoid whipsaw.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3 and L3 levels for 1d
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 6h
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 50-period volume MA
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1w EMA50 trend
            if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
                ema50_slope = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Breakout above H3 with volume confirmation
                    if close[i] > camarilla_h3_1d_aligned[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Breakdown below L3 with volume confirmation
                    if close[i] < camarilla_l3_1d_aligned[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below H3 or opposite signal
            if close[i] < camarilla_h3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above L3 or opposite signal
            if close[i] > camarilla_l3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0