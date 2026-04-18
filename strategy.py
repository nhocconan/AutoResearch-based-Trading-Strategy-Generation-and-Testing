#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 12h EMA trend filter.
# Williams Alligator identifies trend phases via smoothed median lines (Jaw, Teeth, Lips).
# Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA.
# Combined with 12h EMA trend filter to avoid counter-trend trades.
# Works in bull markets (Bull Power > 0, price above Jaw, above 12h EMA) and bear markets
# (Bear Power < 0, price below Jaw, below 12h EMA). Low trade frequency expected.

name = "6h_WilliamsAlligator_ElderRay_12hEMA"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Williams Alligator and Elder Ray (EMA13)
    df_1d = get_htf_data(prices, '1d')
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of median price
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: smoothed = (prev * (period-1) + current) / period
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
                else:
                    result[i] = np.nan
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align Alligator lines and Elder Ray to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment check
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_align = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_align = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Bullish alignment AND Bull Power > 0 AND price above 12h EMA34
            if bullish_align and bull_power_aligned[i] > 0 and close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND Bear Power < 0 AND price below 12h EMA34
            elif bearish_align and bear_power_aligned[i] < 0 and close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment OR Bull Power <= 0 OR price below 12h EMA34
            if (not bullish_align) or (bull_power_aligned[i] <= 0) or (close[i] <= ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment OR Bear Power >= 0 OR price above 12h EMA34
            if (not bearish_align) or (bear_power_aligned[i] >= 0) or (close[i] >= ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals