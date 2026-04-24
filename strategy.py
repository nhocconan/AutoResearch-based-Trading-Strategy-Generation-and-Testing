#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for trend direction.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
- Trend filter: 1w EMA50 slope > 0 for bullish bias, < 0 for bearish bias.
- Entry: Long when Lips cross above Teeth AND price > Jaw AND 1w EMA50 up.
         Short when Lips cross below Teeth AND price < Jaw AND 1w EMA50 down.
- Exit: Opposite Alligator cross or 1w EMA50 slope flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false signals).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
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
        result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 and its slope
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Slope: difference between current and previous EMA50
    ema50_slope_1w = np.diff(ema50_1w, prepend=ema50_1w[0])
    
    # Align 1w EMA50 and slope to 1d
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope_1w)
    
    # Williams Alligator on 1d
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # shift right by 8
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # EMA50 warmup, volume MA, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_slope_1w_aligned[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1w EMA50 slope
        bullish_trend = ema50_slope_1w_aligned[i] > 0
        bearish_trend = ema50_slope_1w_aligned[i] < 0
        
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        prev_lips = lips[i-1]
        prev_teeth = teeth[i-1]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish: Lips cross above Teeth AND price > Jaw AND bullish trend
                if (curr_lips > curr_teeth and prev_lips <= prev_teeth and 
                    curr_close > curr_jaw and bullish_trend):
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips cross below Teeth AND price < Jaw AND bearish trend
                elif (curr_lips < curr_teeth and prev_lips >= prev_teeth and 
                      curr_close < curr_jaw and bearish_trend):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Lips cross below Teeth OR bearish trend flip
            if (curr_lips < curr_teeth and prev_lips >= prev_teeth) or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips cross above Teeth OR bullish trend flip
            if (curr_lips > curr_teeth and prev_lips <= prev_teeth) or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0