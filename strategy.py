#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: 1d strategy using KAMA direction filter with RSI extremes and choppiness regime filter.
# Long when KAMA trending up, RSI < 30, and choppy market (CHOP > 61.8).
# Short when KAMA trending down, RSI > 70, and choppy market (CHOP > 61.8).
# Exit when RSI returns to neutral zone (40-60) or chop regime ends.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 15-25 trades/year (60-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close_s.iloc[9]  # seed value
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Calculate Choppiness Index (CHOP) using weekly data as HTF regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr = np.zeros_like(close_1w)
    atr[13] = np.mean(tr[:14])  # seed
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Max/Min close over 14 periods
    max_hc = np.zeros_like(close_1w)
    min_lc = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        max_hc[i] = np.max(high_1w[i-13:i+1])
        min_lc[i] = np.min(low_1w[i-13:i+1])
    # For first 13 periods, use available data
    for i in range(14):
        max_hc[i] = np.max(high_1w[:i+1])
        min_lc[i] = np.min(low_1w[:i+1])
    
    # Chop = 100 * log10(sum(atr) / (max_hc - min_lc)) / log10(14)
    sum_atr = np.zeros_like(close_1w)
    for i in range(14, len(atr)):
        sum_atr[i] = np.sum(atr[i-13:i+1])
    for i in range(14):
        sum_atr[i] = np.sum(atr[:i+1])
    
    denominator = max_hc - min_lc
    denominator = np.where(denominator == 0, 1e-10, denominator)  # avoid division by zero
    chop = 100 * np.log10(sum_atr / denominator) / np.log10(14)
    
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: choppy market (CHOP > 61.8)
        choppy = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or chop regime ends
            if rsi[i] >= 40 and rsi[i] <= 60 or not choppy:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or chop regime ends
            if rsi[i] >= 40 and rsi[i] <= 60 or not choppy:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry with regime confirmation
            kama_up = close[i] > kama[i]
            kama_down = close[i] < kama[i]
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            if kama_up and rsi_oversold and choppy:
                position = 1
                signals[i] = 0.25
            elif kama_down and rsi_overbought and choppy:
                position = -1
                signals[i] = -0.25
    
    return signals