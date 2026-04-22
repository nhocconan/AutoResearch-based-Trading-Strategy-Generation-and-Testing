#!/usr/bin/env python3
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
    
    # Load daily data for ATR and ATR ratio (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 7-day and 30-day ATR from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # ATR ratio (ATR(7)/ATR(30)) - volatility filter
    atr_ratio = np.where(atr_30 > 0, atr_7 / atr_30, 0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Load weekly data for Supertrend (trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR for Supertrend (using 10-period)
    tr1w = high_1w - low_1w
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr2w[0] = tr1w[0]
    tr3w[0] = tr1w[0]
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_1w = pd.Series(trw).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    upperband = (high_1w + low_1w) / 2 + 3.0 * atr_1w
    lowerband = (high_1w + low_1w) / 2 - 3.0 * atr_1w
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upperband[i-1]:
            direction[i] = 1
        elif close_1w[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # Align Supertrend direction to daily timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Volume confirmation: 20-day average volume from daily data
    vol_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (Supertrend direction = 1) + Low volatility (ATR ratio < 0.8) + Volume spike
            if (supertrend_aligned[i] == 1 and 
                atr_ratio_aligned[i] < 0.8 and
                volume[i] > 1.5 * vol_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (Supertrend direction = -1) + Low volatility (ATR ratio < 0.8) + Volume spike
            elif (supertrend_aligned[i] == -1 and 
                  atr_ratio_aligned[i] < 0.8 and
                  volume[i] > 1.5 * vol_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend reversal or volatility expansion
            if position == 1:
                # Exit long: Downtrend or high volatility
                if supertrend_aligned[i] == -1 or atr_ratio_aligned[i] >= 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Uptrend or high volatility
                if supertrend_aligned[i] == 1 or atr_ratio_aligned[i] >= 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Supertrend_ATR_Volume_Filter"
timeframe = "1d"
leverage = 1.0