#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-week trend filter and volume confirmation
# We go long when price is above Alligator teeth (green line) with weekly EMA(34) uptrend and volume spike.
# We go short when price is below Alligator teeth with weekly EMA(34) downtrend and volume spike.
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the move.

name = "12h_WilliamsAlligator_WeeklyTrend_Volume"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Williams Alligator (13,8,5) on 12h data
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return sma
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw_raw = smoothed_moving_average(close, 13)
    teeth_raw = smoothed_moving_average(close, 8)
    lips_raw = smoothed_moving_average(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Alligator teeth is the middle line (8-period SMMA)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(teeth))}), teeth)
    
    # Volume spike: current volume > 2.0 * 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        teeth_level = teeth_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price above teeth + weekly uptrend + volume spike
            if (not np.isnan(teeth_level) and close[i] > teeth_level and 
                close[i] > ema34_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price below teeth + weekly downtrend + volume spike
            elif (not np.isnan(teeth_level) and close[i] < teeth_level and 
                  close[i] < ema34_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below teeth OR weekly trend turns down
            if (not np.isnan(teeth_level) and close[i] < teeth_level) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above teeth OR weekly trend turns up
            if (not np.isnan(teeth_level) and close[i] > teeth_level) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals