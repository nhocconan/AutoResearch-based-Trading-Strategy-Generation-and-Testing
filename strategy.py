#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_Volume_WeeklyTrendFilter
Hypothesis: Daily Camarilla R1/S1 breakout with volume confirmation (>1.5x average) and weekly EMA20 trend filter captures strong momentum moves while avoiding false signals in sideways markets. The weekly trend filter ensures we trade with the higher timeframe direction, reducing whipsaws. Designed for low trade frequency (target: 15-25/year) to minimize fee drag in daily timeframe. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.zeros_like(close_1w)
    ema20_1w[0] = close_1w[0]
    alpha = 2.0 / (20 + 1)
    for i in range(1, len(close_1w)):
        ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First day
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels (based on previous day)
    range_val = prev_high - prev_low
    R1 = close + (range_val * 1.1 / 12)
    S1 = close - (range_val * 1.1 / 12)
    
    # Volume filter: current volume > 1.5x 20-day average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    # ATR for stoploss (14-day)
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
    
    for i in range(20, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20 = ema20_1w_aligned[i]
        r1 = R1[i]
        s1 = S1[i]
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
            # Long: price breaks above R1 with volume and weekly uptrend (price > weekly EMA20)
            if price > r1 and vol_ok and price > ema20:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and weekly downtrend (price < weekly EMA20)
            elif price < s1 and vol_ok and price < ema20:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls below S1 (reversal) or breaks below weekly EMA20 (trend change)
            if price < s1 or price < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 (reversal) or breaks above weekly EMA20 (trend change)
            if price > r1 or price > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_Volume_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0