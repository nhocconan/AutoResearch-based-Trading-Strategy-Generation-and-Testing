#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
- Long when Lips > Teeth > Jaw (bullish alignment) AND 1d close > 1d EMA50 AND volume > 1.5 * 20-period average volume
- Short when Lips < Teeth < Jaw (bearish alignment) AND 1d close < 1d EMA50 AND volume > 1.5 * 20-period average volume
- Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
- Uses 12h primary with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Alligator identifies trend emergence; EMA50 filters regime; volume confirms momentum
- Designed to work in both bull (trend following) and bear (trend following) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nansum(source[:length]) / length
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + PRICE) / LENGTH
        for i in range(length, len(source)):
            if not np.isnan(source[i]) and not np.isnan(result[i-1]):
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
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, 8 bars shift
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (future values move to past)
    jaw[:8] = np.nan  # first 8 values become invalid after shift
    
    # Teeth: 8-period SMMA, 5 bars shift
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, 3 bars shift
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average (moderate spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need SMMA with sufficient lookback + shifts + EMA50 + volume MA
    start_idx = max(13+8, 8+5, 5+3, 50) + 20  # SMMA periods + shifts + EMA + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
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
            # Long exit: Alligator alignment breaks (Lips crosses below Teeth OR Teeth crosses below Jaw)
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips crosses above Teeth OR Teeth crosses above Jaw)
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0