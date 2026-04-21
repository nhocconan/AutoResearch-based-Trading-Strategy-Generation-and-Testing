#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_TrendFilter
Hypothesis: In 12h timeframe, price breaking above Camarilla R1 or below S1 with volume confirmation and daily trend filter (price > EMA50) captures strong momentum moves while avoiding false signals in ranging markets. The Camarilla levels provide institutional-grade support/resistance, and the 1d EMA50 ensures alignment with the daily trend. Designed for low trade frequency (target: 12-37/year) to minimize fee drag. Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros_like(close_1d)
    ema50_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC (requires 1d data)
    # Camarilla formulas: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need previous day's values, so we shift the 1d data by 1
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    # First value is invalid (no previous day), set to 0
    prev_close_1d[0] = 0
    prev_high_1d[0] = 0
    prev_low_1d[0] = 0
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_mult = 1.1 / 12
    R1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * camarilla_mult
    S1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * camarilla_mult
    
    # Align Camarilla levels to 12h timeframe (they are valid for the entire day)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
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
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        r1 = R1_12h[i]
        s1 = S1_12h[i]
        vol_ok = volume_filter[i]
        
        # Stoploss: 2.5 * ATR from entry (using 12h ATR)
        # Calculate ATR on the fly for simplicity (using 14-period)
        if i >= 14:
            tr = np.zeros(14)
            for j in range(14):
                idx = i - j
                tr1 = high[idx] - low[idx]
                tr2 = abs(high[idx] - close[idx-1]) if idx > 0 else tr1
                tr3 = abs(low[idx] - close[idx-1]) if idx > 0 else tr1
                tr[j] = max(tr1, max(tr2, tr3))
            atr_val = np.mean(tr)
        else:
            atr_val = 0  # Not enough data, but we skip early anyway
        
        # Stoploss check
        if position == 1 and price < entry_price - 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and daily uptrend (price > 1d EMA50)
            if price > r1 and vol_ok and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and daily downtrend (price < 1d EMA50)
            elif price < s1 and vol_ok and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price falls back below R1 or breaks below daily EMA50 (trend change)
            if price < r1 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S1 or breaks above daily EMA50 (trend change)
            if price > s1 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0