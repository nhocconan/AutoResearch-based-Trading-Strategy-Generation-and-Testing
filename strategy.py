#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike
# Long when: Jaw < Teeth < Lips (bullish alignment) AND Elder Bull Power > 0 AND Volume Spike
# Short when: Jaw > Teeth > Lips (bearish alignment) AND Elder Bear Power < 0 AND Volume Spike
# Williams Alligator (13,8,5 SMAs smoothed) identifies trend direction and avoids ranging markets
# Elder Ray measures bull/bear power relative to EMA13 for confirmation
# Volume spike (2.0x 20-bar MA) ensures breakout validity
# Works in bull (trend alignment + volume) and bear (reverse alignment + volume)
# Timeframe: 12h (primary timeframe as required)
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while capturing strong trends

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Williams Alligator (weekly trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Elder Ray (daily EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1w: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_1w = df_1w['close'].values
    # Jaw: 13-period SMA, smoothed by 8-period SMA
    jaw_raw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean().values
    # Teeth: 8-period SMA, smoothed by 5-period SMA
    teeth_raw = pd.Series(close_1w).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean().values
    # Lips: 5-period SMA, smoothed by 3-period SMA
    lips_raw = pd.Series(close_1w).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for indicators)
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment AND positive Bull Power AND volume spike
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND negative Bear Power AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power turns negative
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power turns positive
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals