#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8-period smoothed), Teeth (EMA8, 5-period smoothed), Lips (EMA5, 3-period smoothed)
# In uptrend: Lips > Teeth > Jaw; in downtrend: Lips < Teeth < Jaw
# Strong alignment + volume spike + 1d EMA34 trend filter = high-probability entries
# Works in bull markets (uptrend alignment) and bear markets (downtrend alignment)
# Discrete sizing 0.30 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Williams Alligator components (using 12h timeframe data)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Jaw: EMA13 smoothed by 8 periods
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA8 smoothed by 5 periods
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA5 smoothed by 3 periods
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (uptrend alignment) AND 1d EMA34 uptrend AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Lips < Teeth < Jaw (downtrend alignment) AND 1d EMA34 downtrend AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth OR Teeth < Jaw) OR price closes below Jaw
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth OR Teeth > Jaw) OR price closes above Jaw
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals