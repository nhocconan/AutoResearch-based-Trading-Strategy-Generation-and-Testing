#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop regime filter for BTC/ETH
# - Uses 1d KAMA (Kaufman Adaptive Moving Average) to detect trend direction
# - 1d RSI(14) for overbought/oversold conditions (long when RSI<40 in uptrend, short when RSI>60 in downtrend)
# - 1w Choppiness Index filter: only trade when CHOP(14) > 61.8 (ranging market) to avoid whipsaws in strong trends
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: 30-100 total trades over 4 years (7-25/year) per 1d strategy guidelines
# - Novelty: Combines adaptive trend (KAMA) with mean reversion (RSI) in choppy markets, works in both bull/bear
# - BTC/ETH focus: Uses weekly HTF for regime, avoids SOL-only bias

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else np.array([0])
    # Handle array dimensions properly
    if len(change) < len(close):
        change = np.pad(change, (10, 0), 'constant', constant_values=(0,))
    if len(volatility) < len(close):
        volatility = np.pad(volatility, (1, 0), 'constant', constant_values=(0,))
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, len(close)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first values
    rsi[:13] = 50  # Neutral before enough data
    
    # 1d ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w Choppiness Index (CHOP) for regime filter from HTF
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr_w[0]
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).sum().values
    max_high_14_w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14_w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14_w = max_high_14_w - min_low_14_w
    # Avoid division by zero
    chop_w = np.where(range_14_w > 0, 100 * np.log10(atr_14_w / range_14_w) / np.log10(14), 50)
    chop_w = np.where(np.isnan(chop_w), 50, chop_w)
    # Align weekly chop to daily timeframe (completed 1w bar only)
    chop_w_aligned = align_htf_to_ltf(prices, df_1w, chop_w)
    chop_regime = chop_w_aligned > 61.8  # True when ranging/markets suitable for mean reversion
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup period for indicators
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR RSI > 70 (overbought)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR RSI < 30 (oversold)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for KAMA trend + RSI mean reversion in choppy market
            # Long: price > KAMA (uptrend) AND RSI < 40 (oversold pullback) AND chop regime
            if close[i] > kama[i] and rsi[i] < 40 and chop_regime[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price < KAMA (downtrend) AND RSI > 60 (overbought pullback) AND chop regime
            elif close[i] < kama[i] and rsi[i] > 60 and chop_regime[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals