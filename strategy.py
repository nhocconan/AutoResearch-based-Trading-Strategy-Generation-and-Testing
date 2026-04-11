#!/usr/bin/env python3
# 12h_1d_cci_vwap_mean_reversion_v1
# Strategy: 12h mean reversion using CCI deviation from VWAP with 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: In ranging markets, price reverts to VWAP; CCI > 100 indicates overbought, < -100 oversold.
# Use 1d ADX < 25 to identify ranging conditions. Target 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_vwap_mean_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for ranging market filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]), np.absolute(low_1d[1:] - close_1d[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr_1d)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d.values)
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=24, min_periods=24).sum()  # 24 periods = 12h * 2
    vwap_denominator = pd.Series(volume).rolling(window=24, min_periods=24).sum()
    vwap = vwap_numerator / vwap_denominator
    
    # CCI(20) calculation
    tp = typical_price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vwap.iloc[i]) or 
            np.isnan(cci.iloc[i]) or np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range filter: only trade when ADX < 25 (ranging market)
        ranging = adx_1d_aligned[i] < 25
        
        # Entry conditions: mean reversion from VWAP
        if ranging and cci.iloc[i] > 100 and close[i] < vwap.iloc[i] and position != -1:
            # Overbought and price below VWAP -> short
            position = -1
            signals[i] = -0.25
        elif ranging and cci.iloc[i] < -100 and close[i] > vwap.iloc[i] and position != 1:
            # Oversold and price above VWAP -> long
            position = 1
            signals[i] = 0.25
        # Exit conditions: CCI returns to neutral zone or trend emerges
        elif position == 1 and (cci.iloc[i] > -50 or adx_1d_aligned[i] >= 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci.iloc[i] < 50 or adx_1d_aligned[i] >= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals