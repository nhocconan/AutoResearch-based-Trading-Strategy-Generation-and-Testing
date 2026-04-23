#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Power confirmation and volume spike.
Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Power > 0 AND volume > 2.0x 20-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Power < 0 AND volume > 2.0x 20-period average.
Exit when Alligator alignment breaks or Elder Power crosses zero.
Uses 12h primary timeframe for lower trade frequency (target: 12-37 trades/year) and 1d HTF for trend confirmation.
Williams Alligator: jaws=SMA(13,8), teeth=SMA(8,5), lips=SMA(5,3). Elder Power = (close - EMA13) * volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator for trend alignment (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Jaws: SMA(13,8)
    jaws_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8,5)
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5,3)
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Calculate 1d Elder Power for confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    elder_power_1d = (close_1d - ema_13_1d) * volume_1d
    elder_power_aligned = align_htf_to_ltf(prices, df_1d, elder_power_1d)
    
    # 20-period volume average for spike filter (LTF)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20)  # Alligator (21), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(elder_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        elder = elder_power_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Alligator alignment
        bullish_alignment = jaw < tooth < lip
        bearish_alignment = jaw > tooth > lip
        
        if position == 0:
            # Long: Bullish alignment AND Elder Power > 0 AND volume spike
            if bullish_alignment and elder > 0 and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND Elder Power < 0 AND volume spike
            elif bearish_alignment and elder < 0 and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: alignment breaks OR Elder Power <= 0
                if not bullish_alignment or elder <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: alignment breaks OR Elder Power >= 0
                if not bearish_alignment or elder >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dElderPower_VolumeSpike"
timeframe = "12h"
leverage = 1.0