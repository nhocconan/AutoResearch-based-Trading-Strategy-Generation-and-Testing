#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from prior completed 1d for structure
# 1d EMA34 provides trend filter to avoid whipsaw in ranging markets
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Williams Alligator is effective in both bull and bear markets when price is outside the mouth,
# and combining with 1d EMA34 and volume confirmation improves signal quality.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 1d data
    # Alligator Jaw: 13-period SMMA, shifted 8 bars
    # Alligator Teeth: 8-period SMMA, shifted 5 bars
    # Alligator Lips: 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMMA, shift 8)
    jaw = pd.Series(close_1d).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (8-period SMMA, shift 5)
    teeth = pd.Series(close_1d).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (5-period SMMA, shift 3)
    lips = pd.Series(close_1d).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator mouth is defined by Jaw (outer), Teeth (middle), Lips (inner)
        # In uptrend: Lips > Teeth > Jaw (green alignment)
        # In downtrend: Jaw > Teeth > Lips (red alignment)
        # We trade when price is outside the Alligator's mouth
        
        if position == 0:
            # Long conditions: price above Lips AND Lips > Teeth > Jaw (green alignment) + price above 1d EMA34 + volume spike
            if (close[i] > lips_aligned[i] and 
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below Jaw AND Jaw > Teeth > Lips (red alignment) + price below 1d EMA34 + volume spike
            elif (close[i] < jaw_aligned[i] and 
                  jaw_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Teeth (middle line) OR price crosses below 1d EMA34
            if not np.isnan(teeth_aligned[i]) and (close[i] < teeth_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Teeth (middle line) OR price crosses above 1d EMA34
            if not np.isnan(teeth_aligned[i]) and (close[i] > teeth_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals