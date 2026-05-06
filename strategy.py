#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Alligator (Jaw/Teeth/Lips) for trend direction
# Long when price > Alligator Lips AND Alligator Teeth > Alligator Jaw (bullish alignment)
# Short when price < Alligator Lips AND Alligator Teeth < Alligator Jaw (bearish alignment)
# Exit when price crosses Alligator Teeth (reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Williams Alligator provides smooth trend filtering with built-in period separation
# Works in bull (price above rising Alligator) and bear (price below falling Alligator)

name = "1d_1wWilliamsAlligator_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data ONCE before loop for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:  # Need sufficient data for Alligator (max period 13)
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on 1w timeframe
    # Jaw: Smoothed Moving Average (SMA) of 13 periods, shifted 8 bars forward
    # Teeth: SMA of 8 periods, shifted 5 bars forward  
    # Lips: SMA of 5 periods, shifted 3 bars forward
    close_series_1w = pd.Series(close_1w)
    jaw_1w = close_series_1w.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1w = close_series_1w.rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1w = close_series_1w.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 1w Alligator lines to 1d timeframe (wait for completed 1w bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Lips AND Teeth > Jaw (bullish alignment)
            if (close[i] > lips_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Lips AND Teeth < Jaw (bearish alignment)
            elif (close[i] < lips_aligned[i] and teeth_aligned[i] < jaw_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Teeth (trend weakening)
            if close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Teeth (trend weakening)
            if close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals