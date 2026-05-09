#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1d ADX trend filter and volume spike
# Long when price > Alligator teeth (red line), ADX > 25 (trending), and volume > 2x average
# Short when price < Alligator teeth, ADX > 25, and volume > 2x average
# Exit when price crosses back below/above teeth or ADX < 20 (trend weakening)
# Uses Alligator for trend direction, ADX for trend strength, volume for conviction
# Designed to capture strong trends in both bull and bear markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_WilliamsAlligator_1dADX_Trend_Volume"
timeframe = "6h"
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
    
    # Calculate 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Alligator lines: smoothed medians with future shifts
    median = (df_1d['high'] + df_1d['low']) / 2
    
    # Jaw (blue line): 13-period smoothed median, shifted 8 bars forward
    jaw = median.rolling(window=13, min_periods=13).median()
    jaw = jaw.shift(8)
    
    # Teeth (red line): 8-period smoothed median, shifted 5 bars forward
    teeth = median.rolling(window=8, min_periods=8).median()
    teeth = teeth.shift(5)
    
    # Lips (green line): 5-period smoothed median, shifted 3 bars forward
    lips = median.rolling(window=5, min_periods=5).median()
    lips = lips.shift(3)
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Calculate 1d ADX for trend strength filter
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(teeth_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > teeth, ADX > 25 (strong trend), volume spike
            if (close[i] > teeth_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < teeth, ADX > 25, volume spike
            elif (close[i] < teeth_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below teeth or ADX < 20 (trend weakening)
            if (close[i] < teeth_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above teeth or ADX < 20
            if (close[i] > teeth_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals