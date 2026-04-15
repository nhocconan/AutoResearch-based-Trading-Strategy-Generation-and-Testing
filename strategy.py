#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop Regime Filter
# Uses Kaufman Adaptive Moving Average (KAMA) to capture trend direction.
# Enters long when price > KAMA and RSI > 50, short when price < KAMA and RSI < 50.
# Filters trades using Choppiness Index (CHOP) to avoid ranging markets.
# Works in both bull and bear markets by adapting to trend strength and avoiding chop.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (10-period ER, 2/30 SC)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # Avoid look-ahead for first 10 bars
    
    volatility = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    sc = (er * (2/30 - 2/10) + 2/10) ** 2
    sc[0] = 0  # First value
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period) on weekly data
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((highest_high - lowest_low) > 0,
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    # Align KAMA, RSI, and Chop to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)  # Using 1w for alignment base
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: price > KAMA and RSI > 50 and Chop < 61.8 (trending)
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] > 50 and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price < KAMA and RSI < 50 and Chop < 61.8 (trending)
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] < 50 and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal or Chop > 61.8 (choppy market)
        elif position == 1 and (close[i] < kama_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0