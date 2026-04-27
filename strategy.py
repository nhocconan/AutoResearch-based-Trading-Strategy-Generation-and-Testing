#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) with volume confirmation.
# Williams Alligator (13,8,5 SMAs with offsets) identifies trend direction and alignment.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low; measures bull/bear strength.
# Volume filter: current volume > 1.5x 20-period average to ensure institutional participation.
# Long when: Alligator aligned bullish (Jaw<Teeth<Lips), Bull Power > 0 and rising, volume confirmation.
# Short when: Alligator aligned bearish (Jaw>Teeth>Lips), Bear Power > 0 and rising, volume confirmation.
# Exit when: Alligator alignment breaks or power fails.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drift.
# Works in bull markets (captures strong uptrends) and bear markets (captures strong downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day EMA34 for higher timeframe trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator: SMAs with specific periods and offsets
    # Jaw: 13-period SMMA, offset 8 bars forward
    # Teeth: 8-period SMMA, offset 5 bars forward  
    # Lips: 5-period SMMA, offset 3 bars forward
    # Using SMA as approximation for SMMA (simple moving average)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values
    bear_power = ema13.values - low
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup for all indicators
    start_idx = 30  # enough for Alligator setup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray power conditions (must be positive and rising)
        bull_power_rising = (i > 0 and bull_power[i] > bull_power[i-1])
        bear_power_rising = (i > 0 and bear_power[i] > bear_power[i-1])
        
        # Optional: 1d EMA34 trend filter (align with higher timeframe)
        ht_filter_long = close[i] > ema34_1d_aligned[i]
        ht_filter_short = close[i] < ema34_1d_aligned[i]
        
        # Long condition: bullish Alligator alignment + rising Bull Power + volume + HTF alignment
        if (bullish_alignment and 
            bull_power[i] > 0 and 
            bull_power_rising and 
            volume_filter[i] and
            ht_filter_long):
            signals[i] = 0.25
            position = 1
        # Short condition: bearish Alligator alignment + rising Bear Power + volume + HTF alignment
        elif (bearish_alignment and 
              bear_power[i] > 0 and 
              bear_power_rising and 
              volume_filter[i] and
              ht_filter_short):
            signals[i] = -0.25
            position = -1
        # Exit conditions: Alligator alignment breaks or power fails
        elif position == 1 and not (bullish_alignment and bull_power[i] > 0 and bull_power_rising):
            signals[i] = 0.0
            position = 0
        elif position == -1 and not (bearish_alignment and bear_power[i] > 0 and bear_power_rising):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_VolumeFilter"
timeframe = "6h"
leverage = 1.0