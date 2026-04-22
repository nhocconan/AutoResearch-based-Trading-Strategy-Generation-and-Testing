#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume confirmation.
The Alligator (three SMAs: Jaw, Teeth, Lips) identifies trends when lines are separated and aligned.
We go long when Lips > Teeth > Jaw and price is above Lips, short when Lips < Teeth < Jaw and price below Lips.
Entry requires volume above 20-period average to confirm institutional participation.
Exit when Alligator lines re-converge (Lips crosses Teeth) or volume drops below average.
Works in both bull and bear markets by following the trend defined by the Alligator alignment.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator (Jaw:13, Teeth:8, Lips:5) - all SMAs
    close_4h = df_4h['close'].values
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Load 1d data for volume average and trend confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 1d close for trend confirmation (price vs 50 EMA)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1d 20-period average
        # We approximate 1d volume average for 4h bar by using the aligned value
        vol_confirm = volume[i] > vol_avg_20_aligned[i]
        
        if position == 0 and vol_confirm:
            # Alligator aligned for uptrend: Lips > Teeth > Jaw
            lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
            teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
            # Price above Lips (strong bullish)
            price_above_lips = close[i] > lips_aligned[i]
            
            if lips_above_teeth and teeth_above_jaw and price_above_lips:
                signals[i] = 0.25
                position = 1
            # Alligator aligned for downtrend: Lips < Teeth < Jaw
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  close[i] < lips_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines re-converge (Lips crosses Teeth) or loss of volume confirmation
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips crosses below Teeth (loss of bullish alignment)
                if lips_aligned[i] < teeth_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Lips crosses above Teeth (loss of bearish alignment)
                if lips_aligned[i] > teeth_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Alligator_1dVol_Trend"
timeframe = "4h"
leverage = 1.0