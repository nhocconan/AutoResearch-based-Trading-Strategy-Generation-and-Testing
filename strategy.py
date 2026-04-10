#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + 1w trend filter
# - Williams Alligator (6h): Jaw (TEMA13), Teeth (TEMA8), Lips (TEMA5) - trend identification
# - Elder Ray (1d): Bull Power = High - EMA13, Bear Power = EMA13 - Low - trend strength
# - 1w trend filter: Close > EMA20 for bullish bias, Close < EMA20 for bearish bias
# - Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND 1w close > 1w EMA20
# - Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND Bear Power > 0 AND 1w close < 1w EMA20
# - Exit: Alligator alignment breaks (Lips crosses Teeth) or Elder Ray power reverses
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Uses multiple timeframes for confluence: 6h for entry timing, 1d for trend strength, 1w for trend direction

name = "6h_1d_1w_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def tema(series, period):
    """Triple Exponential Moving Average"""
    ema1 = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    return 3 * (ema1 - ema2) + ema3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Williams Alligator
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = tema(close, jaw_period)
    teeth = tema(close, teeth_period)
    lips = tema(close, lips_period)
    
    # Calculate 1d Elder Ray components
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power = ema_13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)  # Use 1d as base for alignment since Alligator is 6h
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_bearish = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray power confirmation
        bull_power_positive = bull_power_aligned[i] > 0
        bear_power_positive = bear_power_aligned[i] > 0
        
        # 1w trend filter
        trend_bullish = close_1w_aligned[i] > ema_20_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = alligator_bullish and bull_power_positive and trend_bullish
        short_entry = alligator_bearish and bear_power_positive and trend_bearish
        
        # Exit conditions: Alligator alignment breaks or Elder Ray power reverses
        exit_long = not alligator_bullish or not bull_power_positive
        exit_short = not alligator_bearish or not bear_power_positive
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals