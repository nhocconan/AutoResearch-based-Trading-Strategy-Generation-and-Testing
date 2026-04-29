#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) combination
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) defines trend direction and market state
# Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength
# Long when: Alligator aligned bullish (LIPS > TEETH > JAW) AND Bull Power rising AND price > Alligator JAW
# Short when: Alligator aligned bearish (LIPS < TEETH < JAW) AND Bear Power falling AND price < Alligator JAW
# Uses discrete sizing (0.25) to minimize fee drag. Alligator filters whipsaw, Elder Ray confirms momentum.
# This combination works in both bull (trend following) and bear (counter-trend retracements) markets.

name = "6h_WilliamsAlligator_1dElderRay_BullBearPower_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate 1d Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d  # High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Low - EMA13
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h Williams Alligator (SMMA = Smoothed Moving Average)
    # JAW: 13-period SMMA, shifted 8 bars
    # TEETH: 8-period SMMA, shifted 5 bars  
    # LIPS: 5-period SMMA, shifted 3 bars
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Apply Alligator shifts (JAW shifted 8, TEETH shifted 5, LIPS shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set shifted values to NaN for invalid positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any Alligator line is NaN
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_bull_power = bull_power_1d_aligned[i]
        curr_bear_power = bear_power_1d_aligned[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator loses bullish alignment OR Bull Power turns down
            if not (curr_lips > curr_teeth > curr_jaw) or curr_bull_power < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator loses bearish alignment OR Bear Power turns up
            if not (curr_lips < curr_teeth < curr_jaw) or curr_bear_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator bullish alignment AND Bull Power positive AND price > JAW
            if (curr_lips > curr_teeth > curr_jaw and 
                curr_bull_power > 0 and 
                curr_close > curr_jaw):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish alignment AND Bear Power negative AND price < JAW
            elif (curr_lips < curr_teeth < curr_jaw and 
                  curr_bear_power < 0 and 
                  curr_close < curr_jaw):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals