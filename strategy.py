#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA21 trend filter + volume confirmation
# The Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# In strong trends, the lines are well-separated and aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend).
# We enter when the Lips cross the Teeth in the direction of the trend, confirmed by 1d EMA21 slope and volume spike.
# Exits occur when the Alligator lines re-cross or trend weakens.
# This combines trend-following with momentum confirmation to avoid whipsaws in ranging markets.
# Targets 15-35 trades per year (~60-140 total over 4 years) to minimize fee drag.

name = "6h_WilliamsAlligator_1dEMA21_Volume"
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
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with specific periods
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    # Using EMA as proxy for SMMA with same smoothing effect
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA21 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_slope = ema21_1d[1:] - ema21_1d[:-1]  # slope: positive = uptrend
    ema21_1d_slope = np.concatenate([[0], ema21_1d_slope])  # align length
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema21_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d_slope)
    
    # Volume confirmation: current volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(ema21_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema21_val = ema21_1d_aligned[i]
        ema21_slope = ema21_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Lips cross above Teeth, Alligator aligned bullish (Lips > Teeth > Jaw), volume confirmation, 1d uptrend
            if lips_val > teeth_val and lips_val > jaw_val and teeth_val > jaw_val and vol_conf_val and ema21_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips cross below Teeth, Alligator aligned bearish (Lips < Teeth < Jaw), volume confirmation, 1d downtrend
            elif lips_val < teeth_val and lips_val < jaw_val and teeth_val < jaw_val and vol_conf_val and ema21_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips cross below Teeth or Alligator loses alignment or 1d trend turns down
            if lips_val < teeth_val or lips_val < jaw_val or teeth_val < jaw_val or ema21_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips cross above Teeth or Alligator loses alignment or 1d trend turns up
            if lips_val > teeth_val or lips_val > jaw_val or teeth_val > jaw_val or ema21_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals