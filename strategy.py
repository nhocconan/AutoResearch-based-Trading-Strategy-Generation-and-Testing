#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop filter
# Long when KAMA trending up, RSI > 50, and Choppiness Index > 61.8 (ranging market)
# Short when KAMA trending down, RSI < 50, and Choppiness Index > 61.8 (ranging market)
# Exit when KAMA changes direction or RSI crosses 50
# KAMA adapts to market noise, RSI filters momentum, Chop filter ensures mean-reversion regime
# Works in both bull/bear by focusing on ranging markets where mean reversion prevails
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    er[9:] = change[9:] / np.maximum(volatility[9:], 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already aligned)
    kama_aligned = kama
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan
    
    # Choppiness Index (14) on weekly data
    atr_1w = np.zeros(len(df_1w))
    tr1 = df_1w['high'].values - df_1w['low'].values
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
    tr1[0] = 0
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_hh = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(atr_1w)
    denom = max_hh - min_ll
    chop[13:] = 100 * np.log10(np.sum(atr_1w[13:]) / np.maximum(denom[13:], 1e-10)) / np.log10(14)
    chop[:13] = np.nan
    
    # Align Chop to daily timeframe with 2-bar delay for confirmation
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 10)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up, RSI > 50, Chop > 61.8 (ranging)
            kama_up = i > 0 and kama_aligned[i] > kama_aligned[i-1]
            long_cond = kama_up and (rsi[i] > 50) and (chop_aligned[i] > 61.8)
            # Short conditions: KAMA down, RSI < 50, Chop > 61.8 (ranging)
            kama_down = i > 0 and kama_aligned[i] < kama_aligned[i-1]
            short_cond = kama_down and (rsi[i] < 50) and (chop_aligned[i] > 61.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down or RSI < 50
            kama_down = i > 0 and kama_aligned[i] < kama_aligned[i-1]
            if kama_down or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up or RSI > 50
            kama_up = i > 0 and kama_aligned[i] > kama_aligned[i-1]
            if kama_up or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals