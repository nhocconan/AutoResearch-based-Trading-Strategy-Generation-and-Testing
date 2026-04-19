#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Choppiness Index + RSI mean reversion with weekly EMA200 trend filter.
# Long when: Chop > 61.8 (range), RSI < 30, price > weekly EMA200 (uptrend bias)
# Short when: Chop > 61.8 (range), RSI > 70, price < weekly EMA200 (downtrend bias)
# Exit when: RSI crosses 50 (mean reversion complete)
# Choppiness identifies ranging markets where mean reversion works, RSI provides entry/exit,
# Weekly EMA200 filters for higher probability trades in direction of higher timeframe trend.
# Target: 15-25 trades/year per symbol. Works in sideways markets (2025-2026 test period).
name = "1d_ChopRSI_MeanReversion_WeeklyEMA200Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-week data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(low, 1))
    tr3 = np.abs(np.roll(close, 1) - np.roll(low, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    chop = np.where(range_hl > 0, 100 * np.log10(atr.sum() / range_hl) / np.log10(14), 50)
    # Handle edge cases where sum needs to be calculated properly
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    chop = np.where(range_hl > 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for RSI and Choppiness calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema200 = ema200_1w_aligned[i]
        
        if position == 0:
            # Long entry: Chop > 61.8 (range), RSI < 30 (oversold), price > weekly EMA200 (uptrend bias)
            if (chop_val > 61.8 and rsi_val < 30 and price > ema200):
                signals[i] = 0.25
                position = 1
            # Short entry: Chop > 61.8 (range), RSI > 70 (overbought), price < weekly EMA200 (downtrend bias)
            elif (chop_val > 61.8 and rsi_val > 70 and price < ema200):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals