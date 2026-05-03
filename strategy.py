#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trending vs ranging markets
# In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend): trade breakouts
# In ranging markets: fade extreme deviations from the Alligator midpoint
# 1d EMA50 filter ensures we only trade in alignment with higher timeframe trend
# Volume spike confirms breakout authenticity
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs with specific periods and offsets
    # JAW: 13-period SMMA, offset 8 bars
    # TEETH: 8-period SMMA, offset 5 bars
    # LIPS: 5-period SMMA, offset 3 bars
    # Using EMA as proxy for SMMA (similar smoothing effect)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Alligator midpoint (average of JAW and LIPS)
    alligator_mid = (jaw_values + lips_values) / 2
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(alligator_mid[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Alligator alignment: JAW > TEETH > LIPS = uptrend, reverse = downtrend
        alligator_long = jaw_values[i] > teeth_values[i] > lips_values[i]
        alligator_short = jaw_values[i] < teeth_values[i] < lips_values[i]
        
        # Deviation from Alligator midpoint (normalized by ATR-like measure)
        dev_from_mid = abs(close[i] - alligator_mid[i])
        avg_dev = np.mean(np.abs(close[max(0, i-20):i+1] - alligator_mid[max(0, i-20):i+1])) if i >= 20 else dev_from_mid
        extreme_dev = dev_from_mid > (2.0 * avg_dev) if avg_dev > 0 else False
        
        if position == 0:
            # In trending market: trade breakouts in direction of trend
            # In ranging market: fade extreme deviations
            if alligator_long and volume_spike:
                # Uptrend: buy on breakout above JAW with volume
                if close[i] > jaw_values[i]:
                    signals[i] = 0.25
                    position = 1
            elif alligator_short and volume_spike:
                # Downtrend: sell on breakdown below JAW with volume
                if close[i] < jaw_values[i]:
                    signals[i] = -0.25
                    position = -1
            elif not (alligator_long or alligator_short):  # Ranging market
                # Fade extreme deviations from midpoint
                if close[i] > alligator_mid[i] and extreme_dev and volume_spike:
                    signals[i] = -0.25  # Sell at upper extreme
                    position = -1
                elif close[i] < alligator_mid[i] and extreme_dev and volume_spike:
                    signals[i] = 0.25   # Buy at lower extreme
                    position = 1
        elif position == 1:
            # Exit long: trend reversal OR price reverts to midpoint
            if not alligator_long or close[i] < alligator_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reversal OR price reverts to midpoint
            if not alligator_short or close[i] > alligator_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals