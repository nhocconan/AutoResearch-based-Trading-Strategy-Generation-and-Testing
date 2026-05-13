#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation (1.5x MA20).
# Enters long when price is above Alligator lips (SMMA5) with 1d bullish trend (close > EMA50) and volume > 1.5x MA20.
# Enters short when price is below Alligator lips (SMMA5) with 1d bearish trend (close < EMA50) and volume > 1.5x MA20.
# Exits when price crosses Alligator teeth (SMMA8) in opposite direction.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: Alligator alignment + HTF trend + volume spike.
# Williams Alligator identifies trend presence and direction, while 1d EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.5x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "12h_Williams_Alligator_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CURRENT_VALUE) / LENGTH
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (blue line): SMMA(13, 8) - median price smoothed
    # Teeth (red line): SMMA(8, 5) - median price smoothed
    # Lips (green line): SMMA(5, 3) - median price smoothed
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)  # SMMA(13)
    teeth = smma(median_price, 8)   # SMMA(8)
    lips = smma(median_price, 5)    # SMMA(5)
    
    # Shift the Alligator lines as per Williams Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align 1d EMA50 to 12h timeframe
    # (already done above)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above Alligator lips with 1d bullish trend and volume spike
            if close[i] > lips_shifted[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Alligator lips with 1d bearish trend and volume spike
            elif close[i] < lips_shifted[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Alligator teeth (trend weakening)
            if close[i] < teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Alligator teeth (trend weakening)
            if close[i] > teeth_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals