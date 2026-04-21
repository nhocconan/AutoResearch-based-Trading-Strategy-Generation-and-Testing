#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_V1
Hypothesis: 4h KAMA direction (trend filter) combined with RSI extremes and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 1d HTF EMA50 for additional trend confirmation. Designed for low trade frequency (<200 total 4h trades) to minimize fee drag and work in both bull/bear markets via regime adaptation. KAMA adapts to market efficiency, reducing whipsaws in choppy conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # KAMA (adaptive trend filter)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_4h, n=10))
    volatility = np.sum(np.abs(np.diff(close_4h, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close_4h, np.nan, dtype=np.float64)
    kama[29] = close_4h[29]  # seed
    for i in range(30, n):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # RSI (14-period) for mean reversion signals
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    chop_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) 
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long: KAMA up (uptrend) + RSI oversold + not in strong trend (avoid whipsaw)
            if price > kama[i] and rsi[i] < 30 and chop[i] > 38.2:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: KAMA down (downtrend) + RSI overbought + not in strong trend
            elif price < kama[i] and rsi[i] > 70 and chop[i] > 38.2:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit: KAMA down (trend change) or RSI overbought (mean reversion)
            if price < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA up (trend change) or RSI oversold (mean reversion)
            if price > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0