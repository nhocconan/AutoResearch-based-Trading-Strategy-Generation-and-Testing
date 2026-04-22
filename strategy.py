#!/usr/bin/env python3
"""
Hypothesis: 1-day KAMA direction with RSI and chop filter. Long when KAMA rising, RSI>50, and CHOP>61.8 (range). Short when KAMA falling, RSI<50, and CHOP>61.8.
Exit when conditions reverse. Uses daily timeframe to reduce trade frequency and focus on higher probability setups. KAMA adapts to market noise, RSI filters momentum, and chop filter ensures we trade in ranging markets where mean reversion works. Designed to work in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1-day data for KAMA and RSI - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation (adaptive moving average)
    # ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    # SSC = [ER * (FastSC - SlowSC) + SlowSC]^2
    # KAMA = KAMA[1] + SSC * (Close - KAMA[1])
    # Using 10-period for efficiency, 2/30 for fast/slow SC
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series - close_1d_series.shift(1)).rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama_1d = np.zeros(len(close_1d))
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc.iloc[i]):
            kama_1d[i] = kama_1d[i-1]
        else:
            kama_1d[i] = kama_1d[i-1] + sc.iloc[i] * (close_1d[i] - kama_1d[i-1])
    
    # RSI calculation
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Chopiness Index calculation
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = abs(high - close_1d_series.shift())
    tr3 = abs(low - close_1d_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).sum()
    max_high = high.rolling(14).max()
    min_low = low.rolling(14).min()
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    
    # Align indicators to lower timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # RSI conditions
        rsi_over_50 = rsi_1d_aligned[i] > 50
        rsi_under_50 = rsi_1d_aligned[i] < 50
        
        # Chop filter: chop > 61.8 indicates ranging market
        chop_high = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA rising, RSI>50, chop>61.8
            if kama_rising and rsi_over_50 and chop_high:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI<50, chop>61.8
            elif kama_falling and rsi_under_50 and chop_high:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA falling OR RSI<50
                if not kama_rising or not rsi_over_50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA rising OR RSI>50
                if not kama_falling or not rsi_under_50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0