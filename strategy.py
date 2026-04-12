#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_alligator_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator and Elder Ray
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Alligator: SMAs of median price
    median_price = (df_1w['high'] + df_1w['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = df_1w['close'].ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (df_1w['high'] - ema13)
    bear_power = (df_1w['low'] - ema13)
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Volume filter on 6h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, reverse = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray confirmation
        elder_long = bull_power_aligned[i] > 0
        elder_short = bear_power_aligned[i] < 0
        
        # Entry conditions
        long_signal = alligator_long and elder_long and volume_ok[i]
        short_signal = alligator_short and elder_short and volume_ok[i]
        
        # Exit: Alligator reverses or Elder Ray diverges
        exit_long = not alligator_long or not elder_long
        exit_short = not alligator_short or not elder_short
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals