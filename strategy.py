#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Williams Alligator.
- Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) of median price.
  Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
- Trend filter: Only trade in direction of 1d EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 1.8x 30-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying Alligator bullish alignment in uptrend, in bear via selling bearish alignment in downtrend.
- Uses smoothed moving averages (SMMA) which reduce whipsaw vs regular SMA/EMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator: (high + low) / 2
    median_price = (high + low) / 2
    
    # Get 1d data for EMA50 trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2
    
    # Williams Alligator components (SMMA)
    # Jaw: 13-period SMMA of median price
    jaw_1d = smma(median_price_1d, 13)
    # Teeth: 8-period SMMA of median price
    teeth_1d = smma(median_price_1d, 8)
    # Lips: 5-period SMMA of median price
    lips_1d = smma(median_price_1d, 5)
    
    # Align Alligator components to 4h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8 * 30-period volume MA
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 13)  # EMA50 + volume MA + Alligator Jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or
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
                    # Bullish Alligator alignment: Lips > Teeth > Jaw
                    if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]) and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Bearish Alligator alignment: Lips < Teeth < Jaw
                    if (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]) and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or opposite signal
            if not (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or opposite signal
            if not (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0