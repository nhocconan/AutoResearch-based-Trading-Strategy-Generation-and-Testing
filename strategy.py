#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_kama_rsi_chop
# Uses KAMA (Kaufman Adaptive Moving Average) on 12h to determine trend direction.
# RSI(14) on 12h for overbought/oversold conditions.
# Choppiness Index (14) on 1d as regime filter: only trade when CHOP > 61.8 (ranging market).
# Long when KAMA rising AND RSI < 30 in ranging market.
# Short when KAMA falling AND RSI > 70 in ranging market.
# Exit when RSI crosses 50 (mean reversion).
# Designed for low trade frequency (target: 12-37/year) to minimize fee drift.
# Works in ranging markets via mean reversion and avoids trending markets via chop filter.

name = "12h_1d_kama_rsi_chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA calculation (12h timeframe)
    # ER = |Close - Close[10]| / Sum|Close - Close[1]| over 10 periods
    # SSC = [ER * (Fastest SC - Slowest SC) + Slowest SC]^2
    # KAMA = KAMA[1] + SSC * (Close - KAMA[1])
    # Fastest SC = 2/(2+1) = 0.6667, Slowest SC = 2/(30+1) = 0.0645
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1])), axis=0)[:n-10] if n > 10 else np.array([])
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full(n, np.nan)
    if n > 10:
        kama[10] = close[10]  # seed
        for i in range(11, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on 12h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Choppiness Index (14) on daily
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    # ATR14 = SMA(True Range, 14)
    # Chop = 100 * log10(ATR14 / (highest_high - lowest_low over 14)) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.subtract(high_1d[1:], close_1d[:-1]))
    tr3 = np.abs(np.subtract(low_1d[1:], close_1d[:-1]))
    tr1 = tr1[1:]  # align with tr2/tr3
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is undefined
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    chop = np.concatenate([np.full(13, np.nan), chop])  # first 13 values undefined
    
    # Align 1d Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align KAMA and RSI to 12h (they are already 12h, but ensure alignment)
    # KAMA and RSI are calculated on 12h close, so no alignment needed
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop > 61.8 indicates ranging market (chop > 61.8)
        if chop_aligned[i] <= 61.8:
            # In trending market, do not trade - flatten
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long: KAMA rising AND RSI < 30 (oversold in ranging market)
        if kama[i] > kama[i-1] and rsi[i] < 30 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: KAMA falling AND RSI > 70 (overbought in ranging market)
        elif kama[i] < kama[i-1] and rsi[i] > 70 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: RSI crosses 50 (mean reversion)
        elif position == 1 and rsi[i] >= 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] <= 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals