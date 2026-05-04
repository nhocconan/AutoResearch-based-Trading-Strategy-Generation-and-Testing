#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period smoothed median), Lips (5-period smoothed median)
# In trending markets: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
# In ranging markets: lines intertwine
# Entry: Lips crosses above Teeth with uptrend EMA34 + volume spike = long
# Entry: Lips crosses below Teeth with downtrend EMA34 + volume spike = short
# Works in both bull and bear markets by capturing strong trends with trend/volume filters
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe

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
    
    # Williams Alligator components (using 12h prices)
    median_price = (high + low) / 2.0
    close_s = pd.Series(close)
    median_s = pd.Series(median_price)
    
    # Jaw: 13-period smoothed median, shifted 8 bars
    jaw = median_s.rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period smoothed median, shifted 5 bars
    teeth = median_s.rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period smoothed median, shifted 3 bars
    lips = median_s.rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips crosses above Teeth AND 1d EMA34 uptrend AND volume spike
            if lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips crosses below Teeth AND 1d EMA34 downtrend AND volume spike
            elif lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Lips crosses below Jaw OR price closes below EMA13 (using close as proxy for trend)
            ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
            if lips[i] < jaw[i] or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Lips crosses above Jaw OR price closes above EMA13
            ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
            if lips[i] > jaw[i] or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals