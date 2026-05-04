#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Choppiness Regime + Volume Confirmation
# Williams Alligator identifies trend direction via smoothed medians (Jaw/Teeth/Lips).
# Choppiness Index filters ranging markets (CHOP > 61.8 = range, < 38.2 = trend).
# Volume confirmation ensures conviction. Designed for 12-37 trades/year on 12h to minimize fee drag.
# Works in bull markets via long when Lips > Teeth > Jaw in trending regime.
# Works in bear markets via short when Lips < Teeth < Jaw in trending regime.

name = "12h_WilliamsAlligator_1dChop_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    # Calculate 1d ATR(14) using Wilder's smoothing
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[13] = np.mean(tr[:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1d Choppiness Index
    sum_atr_14 = np.zeros_like(atr_14_1d)
    for i in range(13, len(atr_14_1d)):
        if i == 13:
            sum_atr_14[i] = np.sum(atr_14_1d[:14])
        else:
            sum_atr_14[i] = sum_atr_14[i-1] + atr_14_1d[i] - atr_14_1d[i-14]
    
    hh_14 = np.zeros_like(high_1d)
    ll_14 = np.zeros_like(low_1d)
    for i in range(13, len(high_1d)):
        hh_14[i] = np.max(high_1d[i-13:i+1])
        ll_14[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = 100 * np.log10(sum_atr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_1d[:13] = np.nan
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1w data for higher timeframe trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Williams Alligator components
    median_price = (high + low + close) / 3
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume confirmation: 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND trending regime (CHOP < 38.2) AND 1w uptrend AND volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                chop_aligned[i] < 38.2 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND trending regime (CHOP < 38.2) AND 1w downtrend AND volume spike
            elif (lips[i] < teeth[i] < jaw[i] and 
                  chop_aligned[i] < 38.2 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator reverses OR chop becomes too high (range) OR 1w trend turns down
            if (lips[i] < teeth[i] or 
                chop_aligned[i] > 61.8 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator reverses OR chop becomes too high (range) OR 1w trend turns up
            if (lips[i] > teeth[i] or 
                chop_aligned[i] > 61.8 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals