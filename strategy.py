#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Buy when price breaks above Camarilla R1 level (bullish breakout) with volume confirmation and 12h uptrend (price > EMA34); sell when price breaks below S1 level (bearish breakout) with volume confirmation and 12h downtrend (price < EMA34). Uses daily Camarilla levels for structure, volume to confirm participation, and 12h EMA34 for trend filter to avoid counter-trend trades. Designed for low trade frequency (<50/year) to minimize fee drag in 4h timeframe. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 34:
            ema34_1d[i] = np.mean(close_1d[:i+1])
        else:
            ema34_1d[i] = np.mean(close_1d[i-34:i])
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            R1[i] = np.nan
            S1[i] = np.nan
        else:
            # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
            R1[i] = close_1d_prev[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
            S1[i] = close_1d_prev[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and 12h uptrend (price > daily EMA34)
            if price > r1 and vol_ok and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and 12h downtrend (price < daily EMA34)
            elif price < s1 and vol_ok and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below S1 (broken support) or breaks below EMA34 (trend change)
            if price < s1 or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 (broken resistance) or breaks above EMA34 (trend change)
            if price > r1 or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0