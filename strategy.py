#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d Elder Ray trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND volume > 1.5x average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND volume > 1.5x average.
Exit when Alligator alignment breaks or Elder Ray power reverses.
Uses 4h timeframe for higher trade frequency with tight entry conditions to avoid fee drag.
1d Elder Ray provides trend filter via bull/bear power. Volume confirmation ensures high-conviction signals.
Target: 75-200 trades over 4 years (19-50/year).
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
    
    # Load 4h data for Williams Alligator - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Williams Alligator on 4h
    # Jaw (blue line): 13-period SMMA smoothed 8 bars ahead
    # Teeth (red line): 8-period SMMA smoothed 5 bars ahead
    # Lips (green line): 5-period SMMA smoothed 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close_4h, 13)
    teeth = smma(close_4h, 8)
    lips = smma(close_4h, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # jaw shifted 8 bars ahead
    teeth = np.roll(teeth, 5) # teeth shifted 5 bars ahead
    lips = np.roll(lips, 3)   # lips shifted 3 bars ahead
    
    # Load 1d data for Elder Ray - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray on 1d
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Alligator bullish alignment (jaws < teeth < lips) AND bull power > 0 AND volume spike
            if (jaw_val < teeth_val and teeth_val < lips_val and 
                bull_power_val > 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment (jaws > teeth > lips) AND bear power < 0 AND volume spike
            elif (jaw_val > teeth_val and teeth_val > lips_val and 
                  bear_power_val < 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment breaks OR bull power turns negative
                if not (jaw_val < teeth_val and teeth_val < lips_val) or bull_power_val <= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment breaks OR bear power turns positive
                if not (jaw_val > teeth_val and teeth_val > lips_val) or bear_power_val >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1dElderRay_Volume"
timeframe = "4h"
leverage = 1.0