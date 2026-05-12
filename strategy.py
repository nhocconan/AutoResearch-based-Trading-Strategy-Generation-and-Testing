#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Trend_Confirmation
Hypothesis: Williams Alligator on daily timeframe with weekly trend filter to filter false signals in both bull and bear markets.
Enters long when price is above Alligator's lips (fast SMA) with bullish alignment (lips > teeth > jaw) and weekly uptrend.
Enters short when price is below Alligator's lips with bearish alignment (lips < teeth < jaw) and weekly downtrend.
Uses weekly timeframe for trend filter to avoid counter-trend trades. Designed for low trade frequency (10-25/year) to minimize fee drag.
"""

name = "1d_WilliamsAlligator_Trend_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs"""
    # Jaw: 13-period SMMA (smoothed moving average) - using SMA as approximation
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate Alligator on daily data
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Alligator warmup (13 periods)
        # Get aligned values for current daily bar
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)[i]
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or 
            np.isnan(lips_val) or np.isnan(ema50_aligned)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above lips + bullish alignment (lips > teeth > jaw) + weekly uptrend
            if (close[i] > lips_val and 
                lips_val > teeth_val and 
                teeth_val > jaw_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below lips + bearish alignment (lips < teeth < jaw) + weekly downtrend
            elif (close[i] < lips_val and 
                  lips_val < teeth_val and 
                  teeth_val < jaw_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below lips or alignment breaks down
            if (close[i] < lips_val or lips_val <= teeth_val or teeth_val <= jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above lips or alignment breaks up
            if (close[i] > lips_val or lips_val >= teeth_val or teeth_val >= jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals