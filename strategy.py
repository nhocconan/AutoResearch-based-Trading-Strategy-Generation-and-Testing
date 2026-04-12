#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_kama_rsichop_v1
# Uses weekly KAMA trend direction (2-period EMA ratio) for bias, combined with
# daily RSI(14) and Choppiness Index(14) for mean-reversion entries.
# Long when: weekly KAMA up, daily RSI < 30, and Choppiness > 61.8 (ranging market).
# Short when: weekly KAMA down, daily RSI > 70, and Choppiness > 61.8.
# Exits when RSI returns to 50 (mean reversion) or Choppiness < 38.2 (trending market).
# Designed for low trade frequency (target: 10-20 trades/year) with high win rate
# in ranging markets, avoiding trend-following whipsaws in strong trends.

name = "1d_1w_kama_rsichop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA trend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (Kaufman Adaptive Moving Average)
    close_1w = df_1w['close'].values
    # Efficiency Ratio: |change| / sum(|abs change|)
    change = np.abs(np.diff(close_1w))
    abs_change = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    er[1:] = change[1:] / (np.abs(np.diff(close_1w[:-1])) + 1e-10)  # Avoid div by zero
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # KAMA trend: slope of KAMA over 2 periods
    kama_trend = np.zeros_like(kama)
    kama_trend[2:] = (kama[2:] - kama[:-2]) / kama[:-2]
    
    # Align weekly KAMA trend to daily timeframe
    kama_trend_aligned = align_htf_to_ltf(prices, df_1w, kama_trend)
    
    # Daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Daily Choppiness Index(14)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                               np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0,
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14),
                    50)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_trend_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: KAMA up, RSI oversold, choppy market
        if (kama_trend_aligned[i] > 0 and rsi[i] < 30 and chop[i] > 61.8 and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: KAMA down, RSI overbought, choppy market
        elif (kama_trend_aligned[i] < 0 and rsi[i] > 70 and chop[i] > 61.8 and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: RSI mean reversion or trending market
        elif position == 1 and (rsi[i] >= 50 or chop[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] <= 50 or chop[i] < 38.2):
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