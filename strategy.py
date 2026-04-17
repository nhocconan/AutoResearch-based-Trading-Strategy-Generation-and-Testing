#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 12h trend filter.
# Uses Williams Alligator (3 SMAs: Jaw 13, Teeth 8, Lips 5) to identify trend absence.
# Elder Ray measures bull/bear power via EMA13.
# Enters long when: Alligator aligned (Lips>Teeth>Jaw) AND Bull Power > 0 AND price > 12h EMA50.
# Enters short when: Alligator aligned (Lips<Teeth<Jaw) AND Bear Power < 0 AND price < 12h EMA50.
# Designed to capture strong trends while avoiding whipsaws in ranging markets.
# Works in bull markets (strong uptrends) and bear markets (strong downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator components (SMAs)
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price_12h = (high_12h + low_12h) / 2
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price_12h, 13)
    teeth = smma(median_price_12h, 8)
    lips = smma(median_price_12h, 5)
    
    # Calculate Elder Ray (Bull/Bear Power) using EMA13
    close_12h_series = pd.Series(close_12h)
    ema13_12h = close_12h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high_12h - ema13_12h  # Bull Power = High - EMA13
    bear_power = low_12h - ema13_12h   # Bear Power = Low - EMA13
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_12h, teeth)
    lips_6h = align_htf_to_ltf(prices, df_12h, lips)
    bull_power_6h = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_12h, bear_power)
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or
            np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or np.isnan(ema50_12h_6h[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment signals
        alligator_long = (lips_6h[i] > teeth_6h[i]) and (teeth_6h[i] > jaw_6h[i])  # Lips > Teeth > Jaw
        alligator_short = (lips_6h[i] < teeth_6h[i]) and (teeth_6h[i] < jaw_6h[i])  # Lips < Teeth < Jaw
        
        # Elder Ray signals
        bull_power_positive = bull_power_6h[i] > 0
        bear_power_negative = bear_power_6h[i] < 0
        
        # Price relative to 12h EMA50
        price_above_ema = close[i] > ema50_12h_6h[i]
        price_below_ema = close[i] < ema50_12h_6h[i]
        
        if position == 0:
            # Long: Alligator aligned up AND Bull Power positive AND price above EMA50
            if alligator_long and bull_power_positive and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND Bear Power negative AND price below EMA50
            elif alligator_short and bear_power_negative and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bear Power turns negative
            if not alligator_long or bear_power_6h[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bull Power turns positive
            if not alligator_short or bull_power_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_12hEMA50"
timeframe = "6h"
leverage = 1.0