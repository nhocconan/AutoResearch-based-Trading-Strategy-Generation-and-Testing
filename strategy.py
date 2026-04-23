#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with 12h volume confirmation.
Long when: Alligator bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume > 1.5x 24-period average.
Short when: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume > 1.5x 24-period average.
Exit when Alligator crosses in opposite direction (jaw/teeth/lips reorder) OR volume drops below average.
Uses 6h timeframe to target ~15-35 trades/year, avoiding fee drag while capturing sustained trends.
Williams Alligator (SMAs with offsets) identifies trend structure, Elder Ray measures bull/bear power behind moves,
volume confirmation filters weak breakouts. Works in both bull (strong uptrends) and bear (strong downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for volume confirmation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 24-period volume average on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Williams Alligator: Jaw (13-period SMA, 8 bars offset), Teeth (8-period SMA, 5 bars offset), Lips (5-period SMA, 3 bars offset)
    # Calculate on 6h close prices
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_raw[i]) or np.isnan(teeth_raw[i]) or np.isnan(lips_raw[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw = jaw_raw[i]
        teeth = teeth_raw[i]
        lips = lips_raw[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma_val = vol_ma_12h_aligned[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Alligator bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume spike
            if (jaw < teeth and teeth < lips and bull > 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume spike
            elif (jaw > teeth and teeth > lips and bear < 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish OR volume drops below average
                if not (jaw < teeth and teeth < lips) or vol_current < vol_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator turns bullish OR volume drops below average
                if not (jaw > teeth and teeth > lips) or vol_current < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0