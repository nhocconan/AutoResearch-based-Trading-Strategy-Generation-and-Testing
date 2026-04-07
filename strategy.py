#!/usr/bin/env python3
"""
6h Chaikin Money Flow with 12h Trend Filter
Long when CMF > 0.1 and 12h close > 12h EMA50 (bullish momentum)
Short when CMF < -0.1 and 12h close < 12h EMA50 (bearish momentum)
Exit when CMF crosses back through zero
Uses volume to confirm institutional participation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cmf_12h_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Chaikin Money Flow (20) ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where(high == low, 0, mfm)  # Avoid division by zero
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum
    cmf = np.where(vol_sum == 0, 0, cmf)  # Avoid division by zero
    
    # === 12h EMA Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(cmf[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CMF crosses below zero
            if cmf[i] < 0 and cmf[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CMF crosses above zero
            if cmf[i] > 0 and cmf[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish trend: 12h close > 12h EMA50
            # Bearish trend: 12h close < 12h EMA50
            if close[i] > ema_50_aligned[i]:
                # Bullish trend - look for long
                if cmf[i] > 0.1 and cmf[i-1] <= 0.1:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < ema_50_aligned[i]:
                # Bearish trend - look for short
                if cmf[i] < -0.1 and cmf[i-1] >= -0.1:
                    position = -1
                    signals[i] = -0.25
    
    return signals