#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WilliamsAlligator_ElderRay_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend and Elder Ray
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Williams Alligator on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Smoothed median prices
    jaw_period, jaw_shift = 13, 8
    teeth_period, teeth_shift = 8, 5
    lips_period, lips_shift = 5, 3
    
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    median_price = (high_1w + low_1w) / 2
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    
    # Align Alligator lines to daily
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted)
    
    # Elder Ray on weekly
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    power_bull = high_1w - ema13_1w
    power_bear = low_1w - ema13_1w
    
    # Smooth power values
    power_bull_smooth = pd.Series(power_bull).ewm(span=5, adjust=False, min_periods=5).mean().values
    power_bear_smooth = pd.Series(power_bear).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    power_bull_aligned = align_htf_to_ltf(prices, df_1w, power_bull_smooth)
    power_bear_aligned = align_htf_to_ltf(prices, df_1w, power_bear_smooth)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(power_bull_aligned[i]) or 
            np.isnan(power_bear_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: Alligator alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Long: Bullish alignment + positive Bull Power
            if bullish_alignment and power_bull_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + negative Bear Power
            elif bearish_alignment and power_bear_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment or Bear Power turns negative
            if bearish_alignment or power_bear_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment or Bull Power turns positive
            if bullish_alignment or power_bull_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals