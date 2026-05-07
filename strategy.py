#!/usr/bin/env python3
name = "6h_ADX_Alligator_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Alligator lines (SMMA)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Align to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Momentum: 6-period ROC on 6h
    roc = np.zeros(n)
    roc[6:] = (close[6:] - close[:-6]) / close[:-6] * 100
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(13, 6)  # Wait for Alligator and ROC
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + positive momentum
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                trending and roc[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + negative momentum
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  trending and roc[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: alignment breaks or momentum fades
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or roc[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: alignment breaks or momentum fades
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or roc[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s ADX + Alligator momentum strategy
# - Uses daily ADX(25) to filter for trending markets only (works in bull/bear)
# - Alligator (Jaw/Teeth/Lips) provides clear trend direction signals
# - 6-period ROC confirms momentum strength and direction
# - Enters when Alligator shows bullish/bearish alignment with ADX confirmation
# - Exits when alignment breaks or momentum fades
# - Works in both bull (catch uptrends) and bear (catch downtrends) markets
# - Position size 0.25 limits drawdown while capturing trends
# - Targets 50-150 trades over 4 years (12-37/year) to avoid fee drag
# - Novel combination: ADX trend filter + Alligator alignment + ROC momentum
# - Avoids whipsaws by requiring strong trend confirmation (ADX>25)