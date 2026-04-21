#!/usr/bin/env python3
"""
6h_HTF_12h_Donchian20_VolumeSpike_HTFTrend_V1
Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high with volume spike and 12h EMA50 uptrend; enter short when price breaks below 20-period Donchian low with volume spike and 12h EMA50 downtrend. Uses ATR-based trailing stop to manage risk. Designed to capture trends in both bull and bear markets with tight entry conditions to limit fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')  # for 12h EMA50 trend filter
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # for long trailing stop
    lowest_low_since_entry = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above Donchian high with volume spike and 12h EMA50 uptrend
            if price > highest_high[i-1] and vol_ok and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: break below Donchian low with volume spike and 12h EMA50 downtrend
            elif price < lowest_low[i-1] and vol_ok and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry for trailing stop
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            # ATR trailing stop: exit if price drops 2.5*ATR from highest high since entry
            if price < highest_high_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry for trailing stop
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest low since entry
            if price > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_12h_Donchian20_VolumeSpike_HTFTrend_V1"
timeframe = "6h"
leverage = 1.0