#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w trend filter and volume confirmation.
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA34 AND volume > 1.5x average
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA34 AND volume > 1.5x average
# Exit when alignment breaks (Lips <= Teeth for long, Lips >= Teeth for short) OR trend reversal
# Uses 1d timeframe for lower frequency, Alligator for trend strength/alignment, 1w EMA for higher timeframe filter, volume for confirmation.
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
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate SMAs for Williams Alligator on 1d
    # Jaw: SMA(13,8) - 13 period, 8 period shift
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: SMA(8,5) - 8 period, 5 period shift
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: SMA(5,3) - 5 period, 3 period shift
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().shift(1)
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
            # EXIT LONG: Alignment breaks (Lips <= Teeth) OR trend reversal (price < 1w EMA34)
            if lips_1d[i] <= teeth_1d[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alignment breaks (Lips >= Teeth) OR trend reversal (price > 1w EMA34)
            if lips_1d[i] >= teeth_1d[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals