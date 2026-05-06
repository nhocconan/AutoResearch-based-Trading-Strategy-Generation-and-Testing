#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Alligator + 12h momentum + volume confirmation
# - Uses 1d Williams Alligator (SMMA of median price) to determine trend direction
# - Uses 12h ROC(10) for momentum confirmation
# - Uses 12h volume spike for entry confirmation
# - Enters long when price > Alligator jaws and ROC > 0 with volume
# - Enters short when price < Alligator teeth and ROC < 0 with volume
# - Exits when price crosses Alligator lines in opposite direction
# - Designed to capture trends while avoiding whipsaws in ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dWilliamsAlligator_ROC_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate median price for Alligator
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator lines (13, 8, 5 SMMA with 8, 5, 3 shifts)
    jaws = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Shift the lines as per Alligator definition
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 12h timeframe
    jaws_12h = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips)
    
    # 12h ROC(10) for momentum
    roc_period = 10
    roc = np.zeros_like(close)
    for i in range(roc_period, n):
        if close[i-roc_period] != 0:
            roc[i] = (close[i] - close[i-roc_period]) / close[i-roc_period] * 100
    
    # 12h volume filter
    vol_ma_10 = np.zeros_like(volume)
    for i in range(10, n):
        vol_ma_10[i] = np.mean(volume[i-10:i])
    volume_spike = volume > (1.5 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaws_12h[i]) or np.isnan(teeth_12h[i]) or 
            np.isnan(lips_12h[i]) or np.isnan(roc[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > jaws AND ROC > 0 with volume spike
            if close[i] > jaws_12h[i] and roc[i] > 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < teeth AND ROC < 0 with volume spike
            elif close[i] < teeth_12h[i] and roc[i] < 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lips or teeth
            if close[i] < lips_12h[i] or close[i] < teeth_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lips or jaws
            if close[i] > lips_12h[i] or close[i] > jaws_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals