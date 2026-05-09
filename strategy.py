#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator combined with 1w/1d trend filter and volume confirmation
# Long when price is above Alligator's teeth (middle line) with bullish alignment (JAWS < TEETH < LIPS),
# weekly uptrend (price > weekly EMA50), and volume > 1.5x average
# Short when price is below Alligator's teeth with bearish alignment (JAWS > TEETH > LIPS),
# weekly downtrend (price < weekly EMA50), and volume > 1.5x average
# Exit when price crosses the Alligator's teeth or weekly trend changes
# Uses Alligator for market structure, weekly EMA for trend filter, volume for confirmation
# Designed to capture trending moves while avoiding choppy markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_Alligator_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs with future shifts)
    # JAWS: 13-period SMMA shifted 8 bars forward
    # TEETH: 8-period SMMA shifted 5 bars forward  
    # LIPS: 5-period SMMA shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply forward shifts (JAWS +8, TEETH +5, LIPS +3)
    jaws = np.full_like(jaws_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaws) > 8:
        jaws[8:] = jaws_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above teeth, bullish alignment, weekly uptrend, volume spike
            if (close[i] > teeth[i] and 
                jaws[i] < teeth[i] and teeth[i] < lips[i] and  # JAWS < TEETH < LIPS
                close[i] > ema50_1w_aligned[i] and  # Weekly uptrend
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below teeth, bearish alignment, weekly downtrend, volume spike
            elif (close[i] < teeth[i] and 
                  jaws[i] > teeth[i] and teeth[i] > lips[i] and  # JAWS > TEETH > LIPS
                  close[i] < ema50_1w_aligned[i] and  # Weekly downtrend
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses teeth or weekly trend turns down
            if (close[i] <= teeth[i]) or (close[i] <= ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses teeth or weekly trend turns up
            if (close[i] >= teeth[i]) or (close[i] >= ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals