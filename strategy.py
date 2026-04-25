#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Trade 1d timeframe using Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum filter (avoid extremes), and Choppiness Index(14) for regime filter
(only trade when CHOP > 61.8 = ranging market). Enter long when KAMA turns up AND RSI < 70 AND CHOP > 61.8.
Enter short when KAMA turns down AND RSI > 30 AND CHOP > 61.8. Exit on opposite KAMA turn.
Uses discrete sizing 0.25 to balance return and drawdown. Target 10-25 trades/year on 1d timeframe.
KAMA adapts to market noise, reducing false signals. RSI filter avoids overbought/oversold exhaustion.
Chop filter ensures we only trade in ranging regimes where mean reversion works, avoiding strong trends
that cause whipsaws. Designed to work in both bull and bear via regime adaptation.
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
    
    # Get 1w data for HTF trend filter (optional, not used in this version)
    # df_1w = get_htf_data(prices, '1w')
    # close_1w = df_1w['close'].values
    
    # Calculate KAMA(10, 2, 30) - ER=10, fast=2, slow=30
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of abs changes
    # Fix array lengths: change is len(prices)-10, volatility is len(prices)-1
    # We'll compute ER using rolling window approach
    close_s = pd.Series(close)
    change_10 = close_s.diff(10).abs()
    volatility_10 = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change_10 / volatility_10.replace(0, np.nan)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close_s.iloc[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(tr_sum_14 / (max_high_14 - min_low_14)) / log10(14)
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    log_tr_sum = np.log10(tr_sum_14.replace(0, np.nan))
    log_range = np.log10(range_14)
    log_np10 = np.log10(14)
    chop = 100 * (log_tr_sum / log_np10) / (log_range / log_np10)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), Chop (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_values[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # KAMA turning up: current KAMA > previous KAMA
            kama_up = kama[i] > kama[i-1]
            # KAMA turning down: current KAMA < previous KAMA
            kama_down = kama[i] < kama[i-1]
            
            # Long: KAMA up AND RSI < 70 (not overbought) AND Chop > 61.8 (ranging)
            long_setup = kama_up and (rsi_values[i] < 70) and (chop_values[i] > 61.8)
            # Short: KAMA down AND RSI > 30 (not oversold) AND Chop > 61.8 (ranging)
            short_setup = kama_down and (rsi_values[i] > 30) and (chop_values[i] > 61.8)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA turns down
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA turns up
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0