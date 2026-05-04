#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period smoothed SMA, 8-bar shift), Teeth (8-period smoothed SMA, 5-bar shift), Lips (5-period smoothed SMA, 3-bar shift)
# Bullish alignment: Lips > Teeth > Jaw + price above Jaw (uptrend)
# Bearish alignment: Lips < Teeth < Jaw + price below Jaw (downtrend)
# Works in bull markets (strong uptrend with proper Alligator alignment) and bear markets (strong downtrend with proper Alligator alignment)
# Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades
# Volume spike (2.0x 20-period volume EMA) confirms momentum
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_shifted = np.roll(ema50_1w, 1)
    ema50_1w_shifted[0] = np.nan
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_shifted)
    
    # Williams Alligator components (using same timeframe as prices)
    close_s = pd.Series(close)
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = close_s.rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = close_s.rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = close_s.rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish Alligator alignment AND 1w EMA50 uptrend AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > jaw[i] and close[i] > ema50_1w_aligned[i] and 
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish Alligator alignment AND 1w EMA50 downtrend AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < jaw[i] and close[i] < ema50_1w_aligned[i] and 
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price closes below Jaw
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price closes above Jaw
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals