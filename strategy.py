#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Alligator lines: Jaw (EMA13, 8-bar shift), Teeth (EMA8, 5-bar shift), Lips (EMA5, 3-bar shift)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34 AND volume > 1.5x average
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34 AND volume > 1.5x average
# Exit when Alligator lines cross (Lips crosses Teeth) OR trend reversal
# Uses 1d timeframe for lower frequency, Alligator for trend strength, 1w EMA for trend filter, volume for confirmation.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull via trend continuation, bear via faded rallies.

name = "1d_WilliamsAlligator_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Alligator lines: Smoothed Moving Average (SMMA) = EMA with alpha=1/period
    # Jaw: EMA13 of median price, 8-bar shift
    # Teeth: EMA8 of median price, 5-bar shift
    # Lips: EMA5 of median price, 3-bar shift
    median_price_1d = (high_1d + low_1d) / 2
    
    # SMMA calculation using EMA with adjust=False (equivalent to Wilder's smoothing)
    jaw_1d = pd.Series(median_price_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1d = pd.Series(median_price_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1d = pd.Series(median_price_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Set shifted values to NaN for invalid periods
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(lips_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(jaw_1d[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34 AND volume confirmation
            if lips_1d[i] > teeth_1d[i] and teeth_1d[i] > jaw_1d[i] and close[i] > ema34_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34 AND volume confirmation
            elif lips_1d[i] < teeth_1d[i] and teeth_1d[i] < jaw_1d[i] and close[i] < ema34_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips crosses below Teeth (Alligator sleeping) OR trend reversal (price < 1w EMA34)
            if lips_1d[i] <= teeth_1d[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips crosses above Teeth (Alligator sleeping) OR trend reversal (price > 1w EMA34)
            if lips_1d[i] >= teeth_1d[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals