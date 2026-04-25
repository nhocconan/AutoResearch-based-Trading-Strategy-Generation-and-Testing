#!/usr/bin/env python3
"""
1d KAMA Direction + RSI(2) + Chop Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, reducing whipsaws in choppy markets. Combined with short-term RSI for mean reversion entries and a chop regime filter to avoid trending markets, this strategy aims to capture reversals in range-bound conditions while avoiding false signals during strong trends. Works in both bull and bear markets by focusing on mean reversion in ranging regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for regime filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 1d close
    close_1d = df_1d['close'].values
    direction = np.abs(close_1d - np.roll(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0) if len(close_1d) > 1 else 0
    er = np.where(volatility != 0, direction / volatility, 0)
    er[0] = 0  # first value has no prior
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(2) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:2] = 50  # neutral for insufficient data
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on 1w for regime filter
    hl_range_1w = df_1w['high'].values - df_1w['low'].values
    atr_1w = []
    tr_1w = np.maximum(
        hl_range_1w,
        np.maximum(
            np.abs(df_1w['high'].values - np.roll(df_1w['close'].values, 1)),
            np.abs(df_1w['low'].values - np.roll(df_1w['close'].values, 1))
        )
    )
    tr_1w[0] = hl_range_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    sum_tr_14 = atr_1w
    max_h_14 = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    min_l_14 = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / (max_h_14 - min_l_14)) / np.log10(14)
    chop = np.where((max_h_14 - min_l_14) != 0, chop, 50)
    chop[0:13] = 50  # neutral for insufficient data
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade when choppy (CHOP > 61.8)
        in_choppy_regime = chop_val > 61.8
        
        if position == 0:
            # Look for entry signals in choppy regime
            if in_choppy_regime:
                # Long: price below KAMA AND RSI oversold (< 20)
                long_entry = (curr_close < kama_val) and (rsi_val < 20)
                # Short: price above KAMA AND RSI overbought (> 80)
                short_entry = (curr_close > kama_val) and (rsi_val > 80)
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # avoid trending markets
        elif position == 1:
            # Long position management
            # Exit: price crosses above KAMA OR RSI overbought (> 70) OR regime changes to trending
            if (curr_close > kama_val) or (rsi_val > 70) or (chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses below KAMA OR RSI oversold (< 30) OR regime changes to trending
            if (curr_close < kama_val) or (rsi_val < 30) or (chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI2_ChopFilter"
timeframe = "1d"
leverage = 1.0