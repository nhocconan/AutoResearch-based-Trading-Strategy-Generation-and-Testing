#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0) if len(close_1d) > 1 else np.zeros_like(close_1d)
    volatility = np.concatenate([np.zeros(9), volatility])  # align length
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_direction = kama_aligned > np.roll(kama_aligned, 1)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, 50), rsi[14:]])  # pad beginning
    
    # Align RSI to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    atr_12h = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        tr = max(high_12h[i] - low_12h[i], 
                 abs(high_12h[i] - close_12h[i-1]), 
                 abs(low_12h[i] - close_12h[i-1]))
        if i == 1:
            atr_12h[i] = tr
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close_12h)
    lowest_low = np.zeros_like(close_12h)
    for i in range(13, len(close_12h)):
        highest_high[i] = np.max(high_12h[i-13:i+1])
        lowest_low[i] = np.min(low_12h[i-13:i+1])
    
    # Chop = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    sum_atr14 = np.zeros_like(close_12h)
    for i in range(13, len(close_12h)):
        sum_atr14[i] = np.sum(atr_12h[i-13:i+1])
    
    denominator = highest_high - lowest_low
    chop = np.where(denominator != 0, 
                    100 * np.log10(sum_atr14 / denominator) / np.log10(14), 
                    50)
    chop = np.concatenate([np.full(13, 50), chop[13:]])  # pad beginning
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~2 days for 4h
    
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: KAMA up, RSI > 50, Chop > 50 (ranging market)
            if kama_direction[i] and rsi_aligned[i] > 50 and chop_aligned[i] > 50 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: KAMA down, RSI < 50, Chop > 50 (ranging market)
            elif not kama_direction[i] and rsi_aligned[i] < 50 and chop_aligned[i] > 50 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: KAMA down OR RSI < 40
            if not kama_direction[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up OR RSI > 60
            if kama_direction[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA direction with RSI momentum filter in choppy markets on 4h timeframe.
# Long when KAMA is rising (trending up), RSI > 50 (bullish momentum), and Chop > 50 (ranging/volatile market).
# Short when KAMA is falling (trending down), RSI < 50 (bearish momentum), and Chop > 50.
# Uses daily KAMA for trend, daily RSI for momentum, and 12h Chop for regime filter.
# Volume confirmation filters weak signals. Cooldown reduces trade frequency.
# Works in bull markets (KAMA up + RSI > 50) and bear markets (KAMA down + RSI < 50).
# Chop > 50 ensures we trade in volatile/ranging conditions where mean reversion works.