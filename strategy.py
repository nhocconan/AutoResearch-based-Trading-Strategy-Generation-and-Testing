#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + 1d Trend Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and alignment.
Trade when all three lines are aligned (bullish or bearish) with volume confirmation.
Works in both bull and bear markets by following the 1d trend filter. Low trade frequency
(~20-30/year) minimizes fee decay while capturing sustained trends.
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator: SMAs with specific offsets
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    # Calculate SMAs
    sma_close = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(sma_close, jaw_offset)
    jaw[:jaw_offset] = np.nan
    
    sma_close_teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(sma_close_teeth, teeth_offset)
    teeth[:teeth_offset] = np.nan
    
    sma_close_lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(sma_close_lips, lips_offset)
    lips[:lips_offset] = np.nan
    
    # Volume filter: current volume > 1.5x 50-period volume average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw AND price above 1d EMA
            if lips_val > teeth_val > jaw_val and price > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Bearish alignment: Lips < Teeth < Jaw AND price below 1d EMA
            elif lips_val < teeth_val < jaw_val and price < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if Alligator alignment breaks or price crosses below 1d EMA
            if not (lips_val > teeth_val > jaw_val) or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if Alligator alignment breaks or price crosses above 1d EMA
            if not (lips_val < teeth_val < jaw_val) or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_Spike_1dTrend"
timeframe = "12h"
leverage = 1.0