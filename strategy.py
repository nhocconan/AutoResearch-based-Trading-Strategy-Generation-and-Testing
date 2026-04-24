#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Alligator lines.
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) on median price.
  Long when Lips > Teeth > Jaw with volume spike, Short when Lips < Teeth < Jaw with volume spike.
- Trend filter: Only trade in direction of 1d EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 1.8x 30-period volume MA to ensure strong participation.
- Discrete signal size: 0.30 to balance return and drawdown control.
- Target: 80-180 total trades over 4 years (20-45/year) for 4h timeframe.
- Works in bull via buying Alligator alignment in uptrend, in bear via selling Alligator alignment in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate median price for Alligator
    median_price_1d = (high_1d + low_1d) / 2
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_DATA) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(median_price_1d, 13)
    teeth = smma(median_price_1d, 8)
    lips = smma(median_price_1d, 5)
    
    # Align Alligator levels to 4h (each 1d bar = 6x 4h bars for 4h timeframe)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8 * 30-period volume MA
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA50 trend
            if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
                ema50_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Long when Lips > Teeth > Jaw (Alligator alignment up) with volume spike
                    if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) and volume_spike[i]:
                        signals[i] = 0.30
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Short when Lips < Teeth < Jaw (Alligator alignment down) with volume spike
                    if (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) and volume_spike[i]:
                        signals[i] = -0.30
                        position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0