#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA50 trend and Alligator lines.
- Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift).
- Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and price above Teeth.
- Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and price below Teeth.
- Trend filter: Only trade in direction of 1d EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 1.8x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying Alligator bullish alignment in uptrend, in bear via selling bearish alignment in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
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
    
    # Get 1d data for EMA50 trend filter and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator on 1d
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = smma(median_price_1d, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (shift into future)
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price_1d, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price_1d, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h (each 1d bar = 2x 12h bars)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
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
                    # Bullish Alligator alignment: Lips > Teeth > Jaw
                    if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                        close[i] > teeth_aligned[i] and volume_spike[i]):
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Bearish Alligator alignment: Lips < Teeth < Jaw
                    if (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                        close[i] < teeth_aligned[i] and volume_spike[i]):
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks bearish OR price closes below Jaw
            if (lips_aligned[i] < teeth_aligned[i] or 
                teeth_aligned[i] < jaw_aligned[i] or 
                close[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks bullish OR price closes above Jaw
            if (lips_aligned[i] > teeth_aligned[i] or 
                teeth_aligned[i] > jaw_aligned[i] or 
                close[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0