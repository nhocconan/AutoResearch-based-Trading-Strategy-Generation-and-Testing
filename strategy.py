#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 (uptrend filter) AND volume > 1.5 * volume SMA20
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 (downtrend filter) AND volume > 1.5 * volume SMA20
- Exit when Alligator alignment breaks or volume drops below threshold
- Designed to capture strong trending moves with fractal-based trend detection
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's EMA"""
    return pd.Series(source).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (using 1d data)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5 * volume SMA20
    volume_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_sma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 50, 20)  # Need Alligator components, 1w EMA50, and volume SMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND uptrend AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND downtrend AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and downtrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR volume drops below threshold
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR volume drops below threshold
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0