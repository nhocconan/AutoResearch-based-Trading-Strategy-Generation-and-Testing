#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# The Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In strong trends, the SMAs are well-ordered and separated.
# We add 1w EMA50 trend filter to ensure we only trade in the primary trend direction.
# Volume confirmation (1d volume > 1.5x 20-period average) ensures institutional participation.
# This combination works in both bull and bear markets by filtering for strong trends only.
# Targets 15-30 trades per year (~60-120 total over 4 years) to minimize fee drag.

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw(13), Teeth(8), Lips(5)
    # Jaw = 13-period SMMA, Teeth = 8-period SMMA, Lips = 5-period SMMA
    # Using SMA as approximation for SMMA (similar enough for this purpose)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment: 
    # Bullish: Lips > Teeth > Jaw (green alignment)
    # Bearish: Lips < Teeth < Jaw (red alignment)
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # 1w trend filter: EMA50
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_prev = np.roll(ema_50_1w.values, 1)  # previous bar's value to avoid look-ahead
    ema_50_1w_prev[0] = np.nan
    
    # Align 1w EMA50 to 1d
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_prev)
    
    # 1d volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (vol_ma.values * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish alignment + price above 1w EMA50 + volume confirmation
            if alligator_bullish[i] and close[i] > ema_50_1w_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish alignment + price below 1w EMA50 + volume confirmation
            elif alligator_bearish[i] and close[i] < ema_50_1w_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or price crosses below 1w EMA50
            if not alligator_bullish[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or price crosses above 1w EMA50
            if not alligator_bearish[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals