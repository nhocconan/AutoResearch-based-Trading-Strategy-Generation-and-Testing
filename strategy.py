#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_VolumeATRStop_v1
Hypothesis: Daily timeframe Camarilla R1/S1 breakout with volume confirmation and ATR trailing stop.
Uses 1w HTF trend filter (price above/below 20-period HMA) to improve performance in both bull and bear markets.
Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R3, S3
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    r3 = prev_close + rang * 3.0 / 12
    s3 = prev_close - rang * 3.0 / 12
    
    # Align to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 1w data once for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        hma_20_1w = np.full(len(prices), np.nan)
    else:
        close_1w = df_1w['close'].values
        # HMA(20): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 10
        sqrt_n = int(np.sqrt(20))
        wma_half = pd.Series(close_1w).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_20_1w = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
    hma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_20_1w)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss and position sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(hma_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        hma_trend = hma_20_1w_aligned[i]
        
        # Update trailing extremes
        if position == 1:
            highest_since_entry = max(highest_since_entry, price)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, price)
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and HTF uptrend (price > HMA)
            if price > r1_aligned[i] and volume_ok and price > hma_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below S1 with volume and HTF downtrend (price < HMA)
            elif price < s1_aligned[i] and volume_ok and price < hma_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Trailing stop: exit if price drops 2.0 * ATR from highest
            if price < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trailing stop: exit if price rises 2.0 * ATR from lowest
            if price > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_VolumeATRStop_v1"
timeframe = "1d"
leverage = 1.0