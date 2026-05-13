#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray power with 1w trend filter.
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 1w EMA34.
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 1w EMA34.
# Exit on close crossing opposite Alligator teeth line.
# Uses 6h primary timeframe with 1w HTF for major trend alignment and 1d for Elder Ray.
# Williams Alligator identifies trend phases via smoothed medians, Elder Ray measures bull/bear power relative to EMA13.
# Combines trend confirmation (Alligator) with momentum measurement (Elder Ray) for high-conviction entries.
# Designed to work in both bull and bear markets by requiring alignment with 1w trend via EMA34 filter.

name = "6h_Alligator_ElderRay_1wEMA34_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 for Elder Ray power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Williams Alligator lines (smoothed medians)
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead  
    # Lips: 5-period SMMA, shifted 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate median price for Alligator input
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply Alligator shifts (jaw shifted 8, teeth shifted 5, lips shifted 3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w close for major trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1w bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment AND Bull Power > 0 AND price > 1w EMA34
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and bull_power[i] > 0 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment AND Bear Power < 0 AND price < 1w EMA34
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and bear_power[i] < 0 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Alligator teeth (trend weakening)
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Alligator teeth (trend weakening)
            if close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals