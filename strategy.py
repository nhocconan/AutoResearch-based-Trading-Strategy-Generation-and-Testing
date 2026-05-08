#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with daily trend filter and volume confirmation
# We go long when Jaw (13-period SMA) > Teeth (8-period SMA) > Lips (5-period SMA) with daily EMA(34) uptrend and volume spike.
# We go short when Jaw < Teeth < Lips with daily EMA(34) downtrend and volume spike.
# Williams Alligator uses smoothed moving averages to filter noise and identify trends.
# Daily trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation.
# Target: 12-37 trades/year on 12h timeframe to avoid excessive frequency.

name = "12h_WilliamsAlligator_DailyTrend_Volume"
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
    
    # Get daily data once for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components on daily data
    # Using smoothed moving averages (SMMA) as per original Alligator
    daily_close = df_1d['close'].values
    
    # Lips: 5-period SMMA
    lips = pd.Series(daily_close).rolling(window=5, min_periods=5).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(daily_close).rolling(window=8, min_periods=8).mean().values
    # Jaw: 13-period SMMA
    jaw = pd.Series(daily_close).rolling(window=13, min_periods=13).mean().values
    
    # Align Alligator components to 12h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Jaw > Teeth > Lips (Alligator bullish alignment) + daily uptrend + volume spike
            if (jaw_val > teeth_val > lips_val and 
                close[i] > ema34_1d_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Jaw < Teeth < Lips (Alligator bearish alignment) + daily downtrend + volume spike
            elif (jaw_val < teeth_val < lips_val and 
                  close[i] < ema34_1d_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks down OR daily trend turns down
            if not (jaw_val > teeth_val > lips_val) or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks down OR daily trend turns up
            if not (jaw_val < teeth_val < lips_val) or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals