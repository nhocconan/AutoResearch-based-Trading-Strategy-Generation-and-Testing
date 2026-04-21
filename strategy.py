#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter
Hypothesis: Daily Camarilla pivot R1/S1 breakouts with weekly trend filter (price > weekly EMA50) and volume confirmation (>1.5x average) capture institutional breakouts in both bull and bear markets. The weekly EMA50 ensures we trade with the higher timeframe trend, reducing false signals during sideways periods. Volume confirmation ensures breakouts are backed by participation. Designed for low trade frequency (target: 15-25/year) to minimize fee drag in 1d timeframe. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros_like(close_1w)
    ema50_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day: use same values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_factor = (prev_high - prev_low) * 1.1 / 12
    r1 = prev_close + camarilla_factor
    s1 = prev_close - camarilla_factor
    
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
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if NaN in critical values
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        r1_level = r1[i]
        s1_level = s1[i]
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
            # Long: price breaks above R1 with volume and weekly uptrend (price > weekly EMA50)
            if price > r1_level and vol_ok and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and weekly downtrend (price < weekly EMA50)
            elif price < s1_level and vol_ok and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls back below R1 or breaks below weekly EMA50 (trend change)
            if price < r1_level or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S1 or breaks above weekly EMA50 (trend change)
            if price > s1_level or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0