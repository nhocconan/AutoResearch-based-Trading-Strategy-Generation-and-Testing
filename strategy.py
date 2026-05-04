#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (JAW/TEETH/LIPS) identifies trend absence when lines are intertwined.
# Entry on Alligator "awakening" (LIPS crosses JAW/TEETH) in direction of 1w EMA50 trend.
# Volume spike (>1.8 x 50-period EMA) confirms participation, reducing false signals.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via trend-following awakenings and in bear markets via filtered mean-reversion.
# Uses HTF=1w/1d as specified in experiment #123208.

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams Alligator (JAW=13, TEETH=8, LIPS=5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price shifted into future
    # Median price = (high + low) / 2
    median_price_1d = (df_1d['high'] + df_1d['low']) / 2.0
    close_1d = df_1d['close']
    
    # JAW: 13-period SMMA shifted 8 bars ahead
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA shifted 5 bars ahead
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA shifted 3 bars ahead
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: 50-period EMA of volume
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 1.8 x 50-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_50[i])
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        # Alligator signals: LIPS crossing JAW/TEETH indicates trend
        lips_above_jaw = lips_aligned[i] > jaw_aligned[i]
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        lips_below_jaw = lips_aligned[i] < jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        
        if position == 0:
            # Long: LIPS crosses above JAW/TEETH + volume spike + bullish 1w trend
            if (lips_above_jaw and lips_above_teeth and volume_spike and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: LIPS crosses below JAW/TEETH + volume spike + bearish 1w trend
            elif (lips_below_jaw and lips_below_teeth and volume_spike and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: LIPS crosses below TEETH OR 1w trend turns bearish
            if (lips_aligned[i] < teeth_aligned[i] or bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: LIPS crosses above TEETH OR 1w trend turns bullish
            if (lips_aligned[i] > teeth_aligned[i] or bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals