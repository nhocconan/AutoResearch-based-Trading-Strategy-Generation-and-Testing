#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Volume_TrendFilter
Hypothesis: Price breaking above/below Camarilla R1/S1 levels on 12h timeframe with volume confirmation (>1.5x average) and daily trend filter (price > EMA20) captures breakout momentum while avoiding false signals in ranging markets. The daily EMA20 ensures we trade with the higher timeframe trend, reducing whipsaws during pullbacks. Designed for low trade frequency (target: 15-30/year) to minimize fee drag in 12h timeframe. Uses discrete position sizing (0.25) to reduce churn. Works in both bull and bear markets by requiring volume confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data once for EMA20 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 20:
            ema20_1d[i] = np.mean(close_1d[:i+1])
        else:
            ema20_1d[i] = np.mean(close_1d[i-20:i+1])
    
    # Align 1d EMA20 to 12h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Load 1d data for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
        else:
            # Use previous day's OHLC to calculate today's Camarilla levels
            high_prev = high_1d[i-1]
            low_prev = low_1d[i-1]
            close_prev = close_1d[i-1]
            camarilla_r1[i] = close_prev + 1.1 * (high_prev - low_prev) / 12
            camarilla_s1[i] = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align Camarilla levels to 12h timeframe with 1-day delay (previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i+1])
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
            atr[i] = np.mean(tr[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20 = ema20_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
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
            # Long: price breaks above R1 with volume and daily uptrend (price > EMA20)
            if price > r1 and vol_ok and price > ema20:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and daily downtrend (price < EMA20)
            elif price < s1 and vol_ok and price < ema20:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below S1 (reversion) or breaks below EMA20 (trend change)
            if price < s1 or price < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 (reversion) or breaks above EMA20 (trend change)
            if price > r1 or price > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0