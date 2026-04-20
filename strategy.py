# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Keltner_RSI_Trend_V1
Hypothesis: 4h Keltner Channel breakout in direction of 1d EMA50 trend with RSI momentum filter. 
Keltner breakouts capture volatility expansion moves. Trend filter ensures alignment with higher timeframe direction. 
RSI avoids overextended entries. Designed for fewer trades (<50/year) with position size 0.25 to manage drawdown in both bull and bear markets.
"""

name = "4h_Keltner_RSI_Trend_V1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data ONCE before loop for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Keltner Channel (20, 10, 2.0) ===
    # EMA20
    ema20 = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        ema20[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema20[i] = 0.1 * close_4h[i] + 0.9 * ema20[i-1]
    
    # ATR10
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr10 = np.full_like(close_4h, np.nan)
    if len(tr) >= 10:
        atr10[9] = np.mean(tr[:10])
        for i in range(10, len(tr)):
            atr10[i] = 0.1 * tr[i] + 0.9 * atr10[i-1]
    
    # Keltner Bands
    upper_keltner = ema20 + 2.0 * atr10
    lower_keltner = ema20 - 2.0 * atr10
    
    # === 1d Indicator: EMA50 Trend ===
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = 0.02 * close_1d[i] + 0.98 * ema50_1d[i-1]
    
    # === 4h Indicator: RSI(14) ===
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(close_4h, np.nan)
    avg_loss = np.full_like(close_4h, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (13 * avg_gain[i-1] + gain[i]) / 14
            avg_loss[i] = (13 * avg_loss[i-1] + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 4h timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_4h, ema20)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_4h, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_4h, lower_keltner)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above upper Keltner + price > 1d EMA50 + RSI < 70 (not overbought)
            if close[i] > upper_keltner_aligned[i] and close[i] > ema50_1d_aligned[i] and rsi_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Keltner + price < 1d EMA50 + RSI > 30 (not oversold)
            elif close[i] < lower_keltner_aligned[i] and close[i] < ema50_1d_aligned[i] and rsi_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA20 OR RSI > 80 (overbought)
            if close[i] < ema20_aligned[i] or rsi_aligned[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA20 OR RSI < 20 (oversold)
            if close[i] > ema20_aligned[i] or rsi_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals