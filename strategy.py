#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams Alligator with 4h ADX trend filter and volume confirmation
# The Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# In strong trends, the lines are well-separated and aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend).
# We enter when the Lips cross the Teeth in the direction of the trend, confirmed by 4h ADX > 25 and volume spike.
# Exits occur when the Alligator lines re-cross or ADX falls below 20.
# Targets 15-35 trades per year (~60-140 total over 4 years) to minimize fee drain.

name = "1h_WilliamsAlligator_4hADX_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: EMA as proxy for SMMA
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    di_plus = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    di_minus = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # ADX calculation
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        adx_val = adx_aligned[i]
        vol_conf_val = vol_conf[i]
        session_val = session_filter[i]
        
        if position == 0:
            # Enter long: Lips cross above Teeth, Alligator aligned bullish, ADX > 25, volume confirmation, session
            if (lips_val > teeth_val and lips_val > jaw_val and teeth_val > jaw_val and 
                adx_val > 25 and vol_conf_val and session_val):
                signals[i] = 0.20
                position = 1
            # Enter short: Lips cross below Teeth, Alligator aligned bearish, ADX > 25, volume confirmation, session
            elif (lips_val < teeth_val and lips_val < jaw_val and teeth_val < jaw_val and 
                  adx_val > 25 and vol_conf_val and session_val):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Lips cross below Teeth or ADX < 20 or loss of alignment
            if (lips_val < teeth_val or adx_val < 20 or 
                lips_val < jaw_val or teeth_val < jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Lips cross above Teeth or ADX < 20 or loss of alignment
            if (lips_val > teeth_val or adx_val < 20 or 
                lips_val > jaw_val or teeth_val > jaw_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals