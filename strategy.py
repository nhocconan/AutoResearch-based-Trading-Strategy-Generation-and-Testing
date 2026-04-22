#!/usr/bin/env python3
"""
12h Williams Alligator + Elder Ray with Volume Confirmation
Long when: Alligator bullish (green>red>blue) AND Elder Ray bullish (Bull Power>0) AND Volume > 1.5x 20-period average
Short when: Alligator bearish (blue>red>green) AND Elder Ray bearish (Bear Power<0) AND Volume > 1.5x 20-period average
Exit when: Alligator lines cross (trend change) OR volume drops below average
Williams Alligator identifies trend direction using smoothed medians; Elder Ray measures bull/bear power behind the move;
volume confirmation ensures institutional participation. Designed for low frequency by requiring trend alignment + volume.
Works in bull markets (follows Alligator up) and bear markets (follows Alligator down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for Alligator and Elder Ray - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Williams Alligator: SMMA of median price (H+L)/2
    # Jaw (blue): SMMA(13, 8)
    # Teeth (red): SMMA(8, 5)
    # Lips (green): SMMA(5, 3)
    median_price = (df_1w['high'] + df_1w['low']) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price.values, 13)  # Blue line
    teeth = smma(median_price.values, 8)  # Red line
    lips = smma(median_price.values, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Alligator bullish: Lips > Teeth > Jaw (Green > Red > Blue)
            alligator_bullish = (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i])
            # Alligator bearish: Jaw > Teeth > Lips (Blue > Red > Green)
            alligator_bearish = (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i])
            
            # Long: Alligator bullish AND Bull Power positive AND volume confirmed
            if alligator_bullish and bull_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power negative AND volume confirmed
            elif alligator_bearish and bear_power_aligned[i] < 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Exit if Alligator changes direction (trend change)
            current_bullish = (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i])
            current_bearish = (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i])
            
            if position == 1 and not current_bullish:
                exit_signal = True
            elif position == -1 and not current_bearish:
                exit_signal = True
            
            # Exit if volume drops below average (loss of momentum)
            elif volume[i] < vol_ma_20[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0