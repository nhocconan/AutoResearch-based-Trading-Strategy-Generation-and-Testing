#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction + RSI + chop regime
# KAMA adapts to market efficiency: follows price in trends, stays flat in ranges
# RSI(14) for overbought/oversold within trend direction
# Chop filter: avoid trading when market is choppy (CHOP > 61.8)
# Works in bull/bear by following KAMA trend and fading extremes
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend filter and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on daily
    close_s = pd.Series(close_1d)
    change = abs(close_s.diff(10))
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_1d[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc[i] * (close_1d[i] - kama[-1]))
    kama = np.array(kama)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1-period change for KAMA direction
    kama_change = np.diff(kama_1d_aligned, prepend=kama_1d_aligned[0])
    kama_up = kama_change > 0
    kama_down = kama_change < 0
    
    # Calculate chop on daily
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_1d * 14) / (highest_high_1d - lowest_low_1d + 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_1d_aligned < 61.8  # Trending regime
    
    # Load 12h data for RSI
    close = prices['close'].values
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(kama_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_level = kama_1d_aligned[i]
        is_choppy = chop_1d_aligned[i] >= 61.8
        
        if position == 0:
            # Enter long: KAMA up + RSI oversold + not choppy
            if kama_up[i] and rsi[i] < 30 and not is_choppy:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down + RSI overbought + not choppy
            elif kama_down[i] and rsi[i] > 70 and not is_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA down OR RSI overbought
            if not kama_up[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up OR RSI oversold
            if not kama_down[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_ChopRegime"
timeframe = "12h"
leverage = 1.0