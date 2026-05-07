#!/usr/bin/env python3
name = "4h_WilliamsAlligator_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw: 13-period SMMA, shifted 8 bars
    sma13_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma13_1d, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    sma8_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma8_1d, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    sma5_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma5_1d, 3)
    lips[:3] = np.nan
    
    # Align to 4h
    jaw_aligned = align_ltf_to_htf(prices, df_1d, jaw)
    teeth_aligned = align_ltf_to_htf(prices, df_1d, teeth)
    lips_aligned = align_ltf_to_htf(prices, df_1d, lips)
    
    # 1d EMA(34) for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_ltf_to_htf(prices, df_1d, ema34_1d)
    
    # 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 4h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: jaws < teeth < lips (uptrend) or jaws > teeth > lips (downtrend)
        alligator_long = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        alligator_short = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (-80 to -20) + alligator alignment + volume spike + 1d uptrend
            wr_condition = williams_r[i] < -20 and williams_r[i] > -80
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema34_1d_aligned[i] > ema34_1d_aligned[i-20]
            
            if wr_condition and alligator_long and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (-20 to 0) + alligator alignment + volume spike + 1d downtrend
            elif wr_condition and alligator_short and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R overbought or alligator alignment breaks
            if williams_r[i] > -20 or not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R oversold or alligator alignment breaks
            if williams_r[i] < -80 or not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator on 1d to identify trend alignment, combined with Williams %R
# for overbought/oversold conditions on 4h, volume spike confirmation, and 1d EMA trend filter.
# Works in bull markets (buy when alligator aligned up and %R pulls back from oversold)
# and bear markets (sell when alligator aligned down and %R pulls back from overbought).
# Williams %R provides mean-reversion entries within the trend, reducing whipsaw.
# Position size 0.25 targets 20-40 trades/year, avoiding excessive fees.
# Exit when %R reaches extreme or alligator alignment breaks.