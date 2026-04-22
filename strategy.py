#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1h time frame for entry timing and weekly trend filter.
# Long when: price > Alligator's Jaw (TEMA13) + price > Alligator's Teeth (TEMA8) + weekly close > weekly EMA26
# Short when: price < Alligator's Jaw (TEMA13) + price < Alligator's Teeth (TEMA8) + weekly close < weekly EMA26
# Exit when price crosses back through Alligator's Lips (TEMA5)
# Williams Alligator uses smoothed moving averages (SMMA) which act as dynamic support/resistance.
# Works in trending markets (both bull and bear) by aligning with the weekly trend.
# Target: 15-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1h data for Williams Alligator calculation
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    # Jaw (TEMA13): SMMA of median price, period 13
    # Teeth (TEMA8): SMMA of median price, period 8
    # Lips (TEMA5): SMMA of median price, period 5
    median_price_1h = (high_1h + low_1h) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1h = smma(median_price_1h, 13)
    teeth_1h = smma(median_price_1h, 8)
    lips_1h = smma(median_price_1h, 5)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA26 for trend filter
    ema26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align all indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1h, jaw_1h)
    teeth_aligned = align_htf_to_ltf(prices, df_1h, teeth_1h)
    lips_aligned = align_htf_to_ltf(prices, df_1h, lips_1h)
    ema26_aligned = align_htf_to_ltf(prices, df_1w, ema26_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema26_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema26 = ema26_aligned[i]
        
        if position == 0:
            # Long conditions: price > Jaw AND price > Teeth AND weekly close > weekly EMA26
            if price > jaw and price > teeth and close_1w[-1] > ema26:  # close_1w[-1] is latest weekly close
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND price < Teeth AND weekly close < weekly EMA26
            elif price < jaw and price < teeth and close_1w[-1] < ema26:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Lips
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Lips
                if price < lips:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Lips
                if price > lips:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_1hTEMA_1wEMA26_Trend"
timeframe = "1d"
leverage = 1.0