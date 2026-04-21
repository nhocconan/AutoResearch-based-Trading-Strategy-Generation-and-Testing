#!/usr/bin/env python3
"""
4h_HTF_1d_1w_Camarilla_Pivot_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Combine 1d Camarilla R1/S1 levels with 1w trend filter (price > 1w SMA50 for long, < 1w SMA50 for short) + 4h volume spike (>1.5x 20-bar MA) for breakout entries. ATR(14) stoploss (2.0x). Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (target 15-30/year) to overcome fee drag in bear markets while capturing strong breakouts in bull markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for Camarilla levels
    df_1w = get_htf_data(prices, '1w')  # for trend filter
    
    if len(df_1d) < 5 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w Trend Filter (SMA50) ===
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # === 4h Indicators ===
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: break above 1d Camarilla R1 with volume spike and price > 1w SMA50 (bullish trend)
            if price > r1_aligned[i-1] and vol_ok and price > sma50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Camarilla S1 with volume spike and price < 1w SMA50 (bearish trend)
            elif price < s1_aligned[i-1] and vol_ok and price < sma50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal (break below S1)
            if price < close[i-1] - 2.0 * atr[i] or price < s1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal (break above R1)
            if price > close[i-1] + 2.0 * atr[i] or price > r1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_1w_Camarilla_Pivot_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0