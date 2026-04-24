#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
- Long when Lips > Teeth > Jaw (bullish alignment) AND 1d close > 1d EMA50 AND volume > 1.5 * 20-period average volume
- Short when Lips < Teeth < Jaw (bearish alignment) AND 1d close < 1d EMA50 AND volume > 1.5 * 20-period average volume
- Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw)
- Uses 12h primary with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Alligator identifies trend phases; EMA50 filters regime; volume confirms momentum
- Designed to work in both bull (trend following) and bear (trend following) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's smoothing"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components using SMMA
    # Jaw: 13-period SMMA of median price, smoothed 8 bars
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = smma(jaw_raw, 8)
    
    # Teeth: 8-period SMMA of median price, smoothed 5 bars
    teeth_raw = smma(median_price, 8)
    teeth = smma(teeth_raw, 5)
    
    # Lips: 5-period SMMA of median price, smoothed 3 bars
    lips_raw = smma(median_price, 5)
    lips = smma(lips_raw, 3)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average (volume spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need enough data for SMMA calculations: 13+8+5+3+50 = ~79 bars minimum
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: bullish Alligator alignment AND bullish regime AND volume confirmation
            if bullish_alignment and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND bearish regime AND volume confirmation
            elif bearish_alignment and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator bullish alignment breaks (lips crosses below teeth or jaw)
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator bearish alignment breaks (lips crosses above teeth or jaw)
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0