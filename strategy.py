#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw, Teeth, Lips) with 1d EMA34 trend filter and volume confirmation.
# Jaw (blue) = SMA(13, 8), Teeth (red) = SMA(8, 5), Lips (green) = SMA(5, 3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 1.5x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 1.5x 20-period average.
# Exit when Alligator alignment breaks or volume filter fails.
# Designed for 12h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 1d EMA34 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "12h_WilliamsAlligator_1dEMA34_VolumeFilter"
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
    
    # Williams Alligator components
    # Jaw (blue): SMA(13, 8) - period 13, shift 8
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red): SMA(8, 5) - period 8, shift 5
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green): SMA(5, 3) - period 5, shift 3
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator alignment
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index to ensure all indicators are valid
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 34)  # Max of jaw/teeth/lips shifts + EMA period
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment), price > 1d EMA34, volume filter
            long_cond = lips_above_teeth[i] and teeth_above_jaw[i] and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: Lips < Teeth < Jaw (bearish alignment), price < 1d EMA34, volume filter
            short_cond = lips_below_teeth[i] and teeth_below_jaw[i] and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR volume filter fails
            if not (lips_above_teeth[i] and teeth_above_jaw[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR volume filter fails
            if not (lips_below_teeth[i] and teeth_below_jaw[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals