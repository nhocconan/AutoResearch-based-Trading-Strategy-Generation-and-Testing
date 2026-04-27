#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, KAMA adapts to trend strength while RSI filters overextension. 
Choppiness Index regime filter ensures we only trade in trending markets (CHOP < 61.8). 
Weekly trend alignment (price vs 1w EMA50) avoids counter-trend trades. 
Discrete sizing (0.25) reduces fee drag. Target: 30-100 trades over 4 years.
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
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA (adaptive trend)
    close_1d = df_1d['close'].values
    # Efficiency Ratio: |close - close_10| / sum(|diff|) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of absolute changes
    # Pad volatility to match length
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded != 0, change / volatility_padded, 0)
    # Smoothing constants: fastest SC = 2/(2+1)=0.67, slowest SC = 2/(30+1)=0.0645
    sc = (er * 0.665 + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed at period 10
    for i in range(10, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Choppiness Index regime filter (1d)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only allow trades when not strongly ranging
    
    # Align all indicators to primary timeframe (1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need KAMA (10), RSI (14), EMA50 (50), chop (14)
    start_idx = max(10, 14, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        ema50 = ema50_1w_aligned[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            # Determine trend: price above/below KAMA and weekly EMA50
            uptrend = close_val > kama_val and close_val > ema50
            downtrend = close_val < kama_val and close_val < ema50
            
            # RSI filters: avoid overextended conditions
            rsi_not_overbought = rsi_val < 70
            rsi_not_oversold = rsi_val > 30
            
            if uptrend and rsi_not_overbought and chop_ok:
                # Long in uptrend, not overbought, not choppy
                signals[i] = size
                position = 1
                entry_price = close_val
            elif downtrend and rsi_not_oversold and chop_ok:
                # Short in downtrend, not oversold, not choppy
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: trend reversal (price below KAMA) or RSI overextension
            if close_val < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: trend reversal (price above KAMA) or RSI overextension
            if close_val > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0