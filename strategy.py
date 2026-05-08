#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1-day trend filter and volume confirmation
# The Alligator identifies trending vs ranging markets: when Lips > Teeth > Jaw = uptrend, reverse for downtrend.
# We enter trades when the Alligator aligns in a direction AND price confirms with a break of the 8-period high/low.
# Added 1-day EMA(50) trend filter to avoid counter-trend trades and volume spike for confirmation.
# Designed for low frequency (target 50-150 trades over 4 years) with strong trend capture in both bull/bear markets.

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components (13,8,5 SMAs shifted)
    # Jaw: 13-period SMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 8-period high/low for entry confirmation
    high_8 = pd.Series(high).rolling(window=8, min_periods=8).max().values
    low_8 = pd.Series(low).rolling(window=8, min_periods=8).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(high_8[i]) or np.isnan(low_8[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Alligator bullish alignment + price above 8-period high + daily uptrend + volume spike
            if (lips_val > teeth_val > jaw_val and 
                close[i] > high_8[i] and 
                close[i] > ema50_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish alignment + price below 8-period low + daily downtrend + volume spike
            elif (lips_val < teeth_val < jaw_val and 
                  close[i] < low_8[i] and 
                  close[i] < ema50_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR price breaks below 8-period low
            if (lips_val < teeth_val or close[i] < low_8[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR price breaks above 8-period high
            if (lips_val > teeth_val or close[i] > high_8[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals