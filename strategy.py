#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
- Long when Lips > Teeth > Jaw (bullish alignment) and close > 1w EMA50 (bullish HTF trend)
- Short when Lips < Teeth < Jaw (bearish alignment) and close < 1w EMA50 (bearish HTF trend)
- Volume must be > 2.0x 20-period average for conviction to avoid choppy markets
- ATR-based trailing stop: exit when price moves 2.5x ATR against position from extreme
- Uses 1w HTF for trend filter (more stable than 1d) to reduce whipsaw in bear markets
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
- Designed to work in both bull and bear markets via HTF trend filter and Alligator alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(source) < period:
        return np.full(len(source), np.nan)
    result = np.full(len(source), np.nan)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    jaw = smma(jaw, 8)     # Smoothed again by 8 periods
    teeth = smma(close, 8) # Teeth: 8-period SMMA
    teeth = smma(teeth, 5) # Smoothed again by 5 periods
    lips = smma(close, 5)  # Lips: 5-period SMMA
    lips = smma(lips, 3)   # Smoothed again by 3 periods
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: bullish alignment, HTF trend up (close > EMA50), volume spike
            if bullish_alignment and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: bearish alignment, HTF trend down (close < EMA50), volume spike
            elif bearish_alignment and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Long exit: price drops 2.5x ATR from highest high since entry
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price rises 2.5x ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1wEMA50_VolumeSpike_ATRTrailingStop_v1"
timeframe = "12h"
leverage = 1.0