#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with volume confirmation.
# Uses ADX(14) > 25 for trend strength, Alligator (Jaw/Teeth/Lips) for direction,
# and volume spike for confirmation. Works in both bull and bear markets by
# only taking trades when a strong trend is present, avoiding choppy regimes.

name = "6h_ADX_Alligator_VolumeSpike"
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
    
    # Calculate ADX components (primary timeframe)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Williams Alligator (primary timeframe)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume regime: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Alligator conditions: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        is_uptrend = jaw[i] > teeth[i] > lips[i]
        is_downtrend = jaw[i] < teeth[i] < lips[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        # Generate signals
        if position == 0:
            if strong_trend and is_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif strong_trend and is_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on loss of uptrend structure or weak trend
            if not is_uptrend or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on loss of downtrend structure or weak trend
            if not is_downtrend or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals