#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, KAMA (adaptive trend) defines market direction, RSI(14) filters overextension, and Choppiness Index (CHOP) regime filter avoids whipsaws. Enter long when KAMA upward, RSI<50 (not overbought), CHOP>61.8 (ranging/mild trend). Enter short when KAMA downward, RSI>50 (not oversold), CHOP>61.8. Exit on opposite KAMA signal. This combines trend-following with mean-reversion logic in choppy markets, reducing false breaks. Target: 30-80 trades over 4 years (7-20/year) on BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators (KAMA, RSI, CHOP)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:  # Need sufficient data for KAMA and CHOP
        return np.zeros(n)
    
    # --- KAMA (Kaufman Adaptive Moving Average) ---
    # ER = |Net Change| / Sum(|Changes|)
    # Smoothest ER: 2/(fast+1) - 2/(slow+1); we use fast=2, slow=30
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA[i] = KAMA[i-1] + SC * (price[i] - KAMA[i-1])
    close_1d = df_1d['close'].values
    dir_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    vol_1d = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0, keepdims=True)
    # Fix: vol_1d should be rolling sum
    vol_rolling = pd.Series(dir_1d).rolling(window=10, min_periods=1).sum().values
    er_1d = np.where(vol_rolling > 0, dir_1d / vol_rolling, 0)
    sc_1d = (er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # --- RSI(14) on 1d ---
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- Choppiness Index (CHOP) on 1d ---
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    # We use N=14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close_1d, 1))
    tr3 = np.abs(low - np.roll(close_1d, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close_1d[0])
    tr3[0] = np.abs(low[0] - close_1d[0])
    atr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high - min_low
    chop_denom = np.where(chop_denom == 0, 1, chop_denom)  # avoid div by zero
    chop_1d = 100 * (np.log10(atr_sum) - np.log10(chop_denom)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA, RSI, CHOP ready
    start_idx = max(50, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA direction: price above/below KAMA
        kama_up = close[i] > kama_1d_aligned[i]
        kama_down = close[i] < kama_1d_aligned[i]
        
        # RSI filter: not extreme (avoid overextension)
        rsi_not_overbought = rsi_1d_aligned[i] < 50
        rsi_not_oversold = rsi_1d_aligned[i] > 50
        
        # CHOP filter: choppy/ranging market (CHOP > 61.8) or strong trend (CHOP < 38.2)
        # We use CHOP > 61.8 to avoid whipsaws in strong trends
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Long logic: KAMA up, RSI not overbought, choppy market
        if kama_up and rsi_not_overbought and chop_filter:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: KAMA down, RSI not oversold, choppy market
        elif kama_down and rsi_not_oversold and chop_filter:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit: opposite KAMA signal (trend change)
        elif position == 1 and kama_down:
            signals[i] = 0.0
            position = 0
        elif position == -1 and kama_up:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0