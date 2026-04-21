#!/usr/bin/env python3
"""
12h_HTF_1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter (EMA50) and volume confirmation.
Uses ATR-based trailing stop (2.0x ATR) to manage risk. Position size fixed at 0.25.
Targets 12-37 trades/year per symbol by requiring confluence of price level, trend, and volume.
Designed to work in both bull (breakouts with trend) and bear (fade at extremes in range) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d EMA50 trend filter and Camarilla calculation
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 Trend Filter ===
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1d Camarilla Pivot Levels (based on prior day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, R2, S1, S2
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # R2 = Close + (High - Low) * 1.1/6
    # S2 = Close - (High - Low) * 1.1/6
    rang = high_1d - low_1d
    r1 = close_1d + rang * 1.1 / 12
    s1 = close_1d - rang * 1.1 / 12
    r2 = close_1d + rang * 1.1 / 6
    s2 = close_1d - rang * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 12h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + above 1d EMA50 + volume
            if price > r1_aligned[i-1] and price > ema_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short: price breaks below S1 + below 1d EMA50 + volume
            elif price < s1_aligned[i-1] and price < ema_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            # ATR trailing stop: exit if price drops 2.0*ATR from highest high since entry
            if price < highest_high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest low since entry
            if price > lowest_low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0