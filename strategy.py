#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams Alligator + volume confirmation
# Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend phases and exhaustion
# Volume > 1.5x 20-period EMA confirms institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear: Alligator filters chop, Elder Ray confirms momentum, volume validates

name = "6h_ElderRay_Alligator_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Elder Ray components (6h): EMA13, Bull Power, Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Williams Alligator (1d): SMA(13,8), SMA(8,5), SMA(5,3)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Blue line (13,8)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Red line (8,5)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values    # Green line (5,3)
    
    # Alligator alignment
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Alligator signals: 
    # Mouth open (trending): lips > teeth > jaw (bullish) OR lips < teeth < jaw (bearish)
    # Mouth closed (choppy): lines intertwined
    bullish_alligator = (lips_aligned > teeth_aligned) & (teeth_aligned > jaw_aligned)
    bearish_alligator = (lips_aligned < teeth_aligned) & (teeth_aligned < jaw_aligned)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(13, 13)  # Need EMA13 and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish: Bull Power > 0, bullish Alligator alignment, volume spike
            if (bull_power[i] > 0 and 
                bullish_alligator[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: Bear Power < 0, bearish Alligator alignment, volume spike
            elif (bear_power[i] < 0 and 
                  bearish_alligator[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power turns negative OR Alligator mouth closes (teeth crosses lips)
            if (bear_power[i] < 0) or (teeth_aligned[i] <= lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive OR Alligator mouth closes (teeth crosses lips)
            if (bull_power[i] > 0) or (teeth_aligned[i] >= lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals