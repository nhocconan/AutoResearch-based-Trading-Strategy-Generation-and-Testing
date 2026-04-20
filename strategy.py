#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR (14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4-hour momentum (ROC 12) for trend filter
    roc_period = 12
    roc_4h = np.zeros(n, dtype=float)
    for i in range(roc_period, n):
        if prices['close'].iloc[i] == 0 or prices['close'].iloc[i-roc_period] == 0:
            roc_4h[i] = 0.0
        else:
            roc_4h[i] = (prices['close'].iloc[i] / prices['close'].iloc[i-roc_period] - 1) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        atr_1d_val = atr_1d_aligned[i]
        roc_val = roc_4h[i]
        
        # Skip if any value is NaN
        if np.isnan(atr_1d_val) or np.isnan(roc_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1d_val > 0
        
        if position == 0:
            # Long: positive 4h momentum with sufficient volatility
            if roc_val > 0.5 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: negative 4h momentum with sufficient volatility
            elif roc_val < -0.5 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: momentum turns negative
            if roc_val < -0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum turns positive
            if roc_val > 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_ROC_VolumeFilter_V1
# Uses 4-hour ROC (12) for momentum signal
# Filters with daily ATR to avoid low volatility periods
# Enters long when ROC > 0.5%, short when ROC < -0.5%
# Exits when ROC reverses by 0.2% in opposite direction
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_ROC_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0