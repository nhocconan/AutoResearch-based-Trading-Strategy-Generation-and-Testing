#!/usr/bin/env python3
# Hypothesis: 4h 12-hour Supertrend with volume confirmation and ATR-based stoploss
# Long when Supertrend turns green, short when turns red, with volume > 1.5x 20-period average
# Exit on opposite Supertrend signal or when price closes inside the Supertrend band
# Uses Supertrend for trend direction, volume for conviction, ATR for dynamic stop
# Designed to capture medium-term trends with controlled frequency in both bull and bear markets
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Supertrend_Volume_12hTrend_Filter"
timeframe = "4h"
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
    
    # Calculate 12h Supertrend (ATR=10, multiplier=3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    # Calculate ATR for 12h
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean()
    
    # Calculate Supertrend upper and lower bands
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df_12h.index, dtype=float)
    trend = pd.Series(index=df_12h.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df_12h)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            trend.iloc[i] = 1
        else:
            if close.iloc[i] <= supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                trend.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                trend.iloc[i] = 1
    
    # Align Supertrend and trend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend.values)
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Supertrend uptrend (trend=1) and volume confirmation
            if (trend_aligned[i] == 1 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Supertrend downtrend (trend=-1) and volume confirmation
            elif (trend_aligned[i] == -1 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns downtrend (trend=-1) or price closes below Supertrend
            if (trend_aligned[i] == -1 or close[i] < supertrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns uptrend (trend=1) or price closes above Supertrend
            if (trend_aligned[i] == 1 or close[i] > supertrend_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals