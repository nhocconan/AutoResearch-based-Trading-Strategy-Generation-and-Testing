#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction and 6h Elder Ray (Bull/Bear Power) for entry timing.
# In 1d uptrend (Lips > Teeth > Jaw), wait for 6h Bull Power > 0 to go long (buy the dip in uptrend).
# In 1d downtrend (Lips < Teeth < Jaw), wait for 6h Bear Power < 0 to go short (sell the rally in downtrend).
# Volume confirmation ensures momentum validity. Designed for low trade frequency (12-30/year) to minimize fee drag.
# Alligator uses SMAs with specific periods; Elder Ray uses EMA13 and high/low minus EMA13.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Williams Alligator ===
    # Jaw: Blue line, 13-period SMMA smoothed by 8 bars
    # Teeth: Red line, 8-period SMMA smoothed by 5 bars
    # Lips: Green line, 5-period SMMA smoothed by 3 bars
    close_1d = df_1d['close'].values
    # SMMA (Smoothed Moving Average) = EMA with alpha=1/period
    jaw = pd.Series(close_1d).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values  # additional smoothing
    teeth = pd.Series(close_1d).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values  # additional smoothing
    lips = pd.Series(close_1d).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(alpha=1/3, adjust=False, min_periods=3).mean().values  # additional smoothing
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 6h Indicators: Elder Ray (Bull Power / Bear Power) ===
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d Alligator trend
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Lips < Teeth < Jaw
        is_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        is_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # === LONG CONDITIONS ===
        # 1. In 1d uptrend (Alligator aligned up)
        # 2. 6h Bull Power > 0 (bullish momentum)
        if is_uptrend and (bull_power[i] > 0):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1d downtrend (Alligator aligned down)
        # 2. 6h Bear Power < 0 (bearish momentum)
        elif is_downtrend and (bear_power[i] < 0):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_ElderRay_v1"
timeframe = "6h"
leverage = 1.0