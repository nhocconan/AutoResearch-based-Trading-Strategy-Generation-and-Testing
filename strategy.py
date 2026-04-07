#!/usr/bin/env python3
"""
6H ADX Trend Strength with Volume Confirmation and 1D Trend Filter
Long when ADX > 25 (strong trend) and +DI > -DI (bullish) with volume expansion AND 1D EMA trend up
Short when ADX > 25 (strong trend) and -DI > +DI (bearish) with volume expansion AND 1D EMA trend down
Exit when ADX falls below 20 (weakening trend) or trend reverses
Uses ADX to filter for strong trends only, avoiding whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_trend_volume_1d_filter_v1"
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
    
    # === ADX Calculation (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high - np.roll(high, 1)
    minus_dm = np.roll(low, 1) - low
    plus_dm[0] = 0
    minus_dm[0] = 0
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 1D trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX weakening or trend reversal
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX weakening or trend reversal
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong trend (ADX > 25) and volume expansion
            if adx[i] <= 25 or vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Strong trend with volume confirmation AND 1D trend filter
            if plus_di[i] > minus_di[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Strong bullish trend with rising 1D EMA -> long
                position = 1
                signals[i] = 0.25
            elif minus_di[i] > plus_di[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Strong bearish trend with falling 1D EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals