#!/usr/bin/env python3
# 12h_KAMA_RSI_ChopFilter_v2
# Hypothesis: On 12h timeframe, use KAMA direction as primary trend filter,
# RSI(14) for momentum confirmation, and Choppiness Index(14) to filter ranging markets.
# Long when KAMA rising, RSI > 50, and CHOP > 61.8 (ranging market - mean reversion setup).
# Short when KAMA falling, RSI < 50, and CHOP > 61.8.
# Exit when KAMA direction reverses or RSI crosses 50.
# Uses 1d ATR for volatility normalization and 1d EMA34 as higher timeframe trend confirmation.
# Designed for low trade frequency (15-25/year) to minimize fee drag in ranging/low volatility markets.

name = "12h_KAMA_RSI_ChopFilter_v2"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # ===== KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) =====
    # Fast EMA period = 2, Slow EMA period = 30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ===== RSI(14) =====
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== CHOPPINESS INDEX (14) =====
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop = np.where((highest_high - lowest_low) > 0,
                    100 * np.log10(np.sum(tr[-14:]) / (highest_high - lowest_low)) / np.log10(14),
                    50)
    # For efficiency, compute rolling sum of TR
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    
    # ===== HIGHER TIMEFRAME DATA (1d) =====
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d ATR for volatility normalization
    atr_1d = pd.Series(df_1d['high'].values - df_1d['low'].values).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: 1 if rising, -1 if falling
        kama_direction = 1 if kama[i] > kama[i-1] else -1
        
        if position == 0:
            # LONG: KAMA rising, RSI > 50, CHOP > 61.8 (ranging market), price > 1d EMA34
            if (kama_direction == 1 and 
                rsi[i] > 50 and 
                chop[i] > 61.8 and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, CHOP > 61.8 (ranging market), price < 1d EMA34
            elif (kama_direction == -1 and 
                  rsi[i] < 50 and 
                  chop[i] > 61.8 and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 50
            if kama_direction == -1 or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 50
            if kama_direction == 1 or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals