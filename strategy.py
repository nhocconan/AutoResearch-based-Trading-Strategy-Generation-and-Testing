#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator: Jaw (EMA13, 8-bar offset), Teeth (EMA8, 5-bar offset), Lips (EMA5, 3-bar offset)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 1.5x average
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 1.5x average
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) OR trend reversal
# Uses 12h timeframe for lower frequency, Williams Alligator for trend strength, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend continuation, bear via faded rallies.

name = "12h_WilliamsAlligator_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMAs for Williams Alligator on 12h close
    ema5_12h = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema8_12h = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Williams Alligator lines with offsets (as per original formula)
    # Lips: EMA5, 3-bar offset -> shift right by 3
    lips = np.roll(ema5_12h, 3)
    lips[:3] = np.nan  # First 3 values invalid due to offset
    # Teeth: EMA8, 5-bar offset -> shift right by 5
    teeth = np.roll(ema8_12h, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to offset
    # Jaw: EMA13, 8-bar offset -> shift right by 8
    jaw = np.roll(ema13_12h, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to offset
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema34_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR trend reversal (price < 1d EMA34)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR trend reversal (price > 1d EMA34)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals