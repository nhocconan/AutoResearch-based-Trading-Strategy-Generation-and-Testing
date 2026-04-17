#!/usr/bin/env python3
"""
4h_Williams_Alligator_Momentum_v1
Williams Alligator (Jaw=13, Teeth=8, Lips=5) + momentum filter (ROC 5) + volume confirmation.
Long when price > Teeth and ROC > 0 and volume > avg volume. Short when price < Teeth and ROC < 0 and volume > avg volume.
Exit when price crosses Jaw or volume drops below average.
Uses 1d ADX > 25 as trend filter to avoid choppy markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === Williams Alligator (SMA of median price) ===
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-bar
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # 8-bar
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-bar
    
    # === ROC(5) for momentum ===
    roc = np.zeros_like(close)
    roc[5:] = (close[5:] - close[:-5]) / close[:-5]
    
    # === Volume average (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d ADX(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    plus_dm_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm_1d = np.concatenate([[0], plus_dm_1d])
    minus_dm_1d = np.concatenate([[0], minus_dm_1d])
    
    # Smooth TR and DM
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    minus_di_1d = 100 * pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(roc[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price > Teeth, ROC > 0, volume > average, ADX > 25
            if (close[i] > teeth[i] and 
                roc[i] > 0 and 
                volume[i] > vol_ma[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Teeth, ROC < 0, volume > average, ADX > 25
            elif (close[i] < teeth[i] and 
                  roc[i] < 0 and 
                  volume[i] > vol_ma[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price < Jaw OR volume < average
            if (close[i] < jaw[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Jaw OR volume < average
            if (close[i] > jaw[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Momentum_v1"
timeframe = "4h"
leverage = 1.0