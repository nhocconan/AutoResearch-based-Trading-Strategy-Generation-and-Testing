#/usr/bin/env python3
# 12h_WilliamsAlligator_TripleFilter
# Hypothesis: Combines Williams Alligator (Jaw/Teeth/Lips) with price position relative to Teeth, ADX trend strength filter, and volume spike confirmation. Designed to capture strong trending moves while avoiding chop. Works in both bull/bear markets by requiring alignment of multiple filters. Target: 15-30 trades/year on 12h timeframe.

name = "12h_WilliamsAlligator_TripleFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator (longer-term trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Williams Alligator on weekly close
    jaw = pd.Series(df_1w['close'].values).rolling(window=13, min_periods=13).mean().rolling(8, min_periods=8).mean().values
    teeth = pd.Series(df_1w['close'].values).rolling(window=8, min_periods=8).mean().rolling(5, min_periods=5).mean().values
    lips = pd.Series(df_1w['close'].values).rolling(window=5, min_periods=5).mean().rolling(3, min_periods=3).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get daily data for ADX and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # ADX calculation on daily data
    plus_dm = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > (df_1d['low'].values[:-1] - df_1d['low'].values[1:]), 
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > (df_1d['high'].values[1:] - df_1d['high'].values[:-1]), 
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr3 = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection on 12h chart
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods = 12 days on 12h chart
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 14)  # Ensure we have volume MA and ADX data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above teeth, Alligator aligned (jaw < teeth < lips), strong trend (ADX > 25), volume spike
            if (close[i] > teeth_aligned[i] and 
                jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                adx_aligned[i] > 25 and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below teeth, Alligator aligned (jaw > teeth > lips), strong trend (ADX > 25), volume spike
            elif (close[i] < teeth_aligned[i] and 
                  jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                  adx_aligned[i] > 25 and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below teeth or Alligator loses alignment or weak trend
            if (close[i] < teeth_aligned[i] or 
                not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above teeth or Alligator loses alignment or weak trend
            if (close[i] > teeth_aligned[i] or 
                not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals