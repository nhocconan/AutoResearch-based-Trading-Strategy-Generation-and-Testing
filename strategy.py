#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: Daily KAMA trend + RSI extreme + chop regime filter for BTC/ETH/SOL.
# Long: KAMA rising (bullish trend), RSI < 30 (oversold), chop > 61.8 (ranging/mean-reverting conditions).
# Short: KAMA falling (bearish trend), RSI > 70 (overbought), chop > 61.8 (ranging/mean-reverting conditions).
# Exit: Opposite KAMA direction change or RSI crossing 50.
# Uses 1d for primary signals, 1w for chop regime filter (calculated from weekly HTF).
# Target: 20-60 trades/year (80-240 total over 4 years) with low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA(10,2,30) - Adaptive trend indicator
    close_s = pd.Series(close)
    change = close_s.diff(10).abs()
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    sc = sc.fillna(0)
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for mean reversion signals
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Get 1w data for chop regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Chopiness Index(14)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    tr1 = high_1w - low_1w
    tr2 = (high_1w - close_1w.shift()).abs()
    tr3 = (low_1w - close_1w.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    highest_high = high_1w.rolling(window=14, min_periods=14).max()
    lowest_low = low_1w.rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Align HTF chop to daily timeframe (wait for completed 1w bar)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising/falling
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Chop regime filter: > 61.8 = ranging (good for mean reversion)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: KAMA turns bearish OR RSI crosses above 50 (mean reversion complete)
            if not kama_rising or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns bullish OR RSI crosses below 50 (mean reversion complete)
            if not kama_falling or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KAMA rising + RSI oversold (<30) + chop regime (ranging)
            if kama_rising and rsi[i] < 30 and chop_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA falling + RSI overbought (>70) + chop regime (ranging)
            elif kama_falling and rsi[i] > 70 and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals