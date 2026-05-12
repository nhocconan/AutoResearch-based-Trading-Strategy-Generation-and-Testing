#!/usr/bin/env python3
# 12H_WILLIAMS_ALLIGATOR_1W_TREND_FILTER
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# In 1week uptrend (price > 50-period SMA), go long when Lips > Teeth > Jaw (bullish alignment).
# In 1week downtrend (price < 50-period SMA), go short when Lips < Teeth < Jaw (bearish alignment).
# Uses 12h timeframe for entries, 1week for trend filter and Alligator alignment.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_WILLIAMS_ALLIGATOR_1W_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly data for Alligator and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    close_1w = df_1w['close'].values
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_shift, teeth_shift, lips_shift = 8, 5, 3
    
    # Calculate SMMA using Wilder's smoothing (alpha = 1/period)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, jaw_period)
    teeth = smma(close_1w, teeth_period)
    lips = smma(close_1w, lips_period)
    
    # Apply shifts (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    
    # Set NaN for shifted positions that would look ahead
    jaw_shifted[:jaw_shift] = np.nan
    teeth_shifted[:teeth_shift] = np.nan
    lips_shifted[:lips_shift] = np.nan
    
    # 50-period SMA for trend filter
    sma50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted)
    sma50_aligned = align_htf_to_ltf(prices, df_1w, sma50)
    
    # Bullish alignment: Lips > Teeth > Jaw
    bullish_alignment = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    # Bearish alignment: Lips < Teeth < Jaw
    bearish_alignment = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable (50+ shifts)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(sma50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1week uptrend (price > SMA50) + Bullish Alligator alignment
            if (close[i] > sma50_aligned[i] and 
                bullish_alignment[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1week downtrend (price < SMA50) + Bearish Alligator alignment
            elif (close[i] < sma50_aligned[i] and 
                  bearish_alignment[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or alignment breakdown
            if (close[i] <= sma50_aligned[i] or 
                not bullish_alignment[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or alignment breakdown
            if (close[i] >= sma50_aligned[i] or 
                not bearish_alignment[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals