#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: 1d strategy using KAMA trend direction, RSI for momentum, and Choppiness index for regime filtering.
# KAMA adapts to market noise, reducing whipsaws. RSI (14) identifies overbought/oversold conditions.
# Chop filter (>61.8) ensures we only trade in ranging markets where mean reversion works.
# In bear markets (2025+), price tends to revert from extremes in ranging conditions.
# Volume confirmation is omitted to reduce trade frequency; KAMA + RSI + Chop provides sufficient filtering.
# Target: 30-100 total trades over 4 years (7-25/year) by requiring confluence of three filters.
# Primary timeframe: 1d, HTF: 1w for trend context.

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
    
    # 1w HTF data for trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        # Fallback to 1d if insufficient 1w data
        df_1w = get_htf_data(prices, '1d')
    
    # KAMA (Adaptive Moving Average) - 1d close
    close_s = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s - close_s.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)  # Handle div/0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14) - 1d close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Neutral when undefined
    
    # Choppiness Index (14) - 1d
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    atr_14 = (high_s - low_s).rolling(window=14, min_periods=14).sum()
    high_14 = high_s.rolling(window=14, min_periods=14).max()
    low_14 = low_s.rolling(window=14, min_periods=14).min()
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    # Align 1w trend filter (optional: only trade in direction of 1w KAMA)
    if len(df_1w) >= 20:
        close_1w = df_1w['close'].values
        close_1w_s = pd.Series(close_1w)
        change_1w = abs(close_1w_s - close_1w_s.shift(10))
        volatility_1w = abs(close_1w_s - close_1w_s.shift(1)).rolling(window=10, min_periods=10).sum()
        er_1w = change_1w / volatility_1w
        er_1w = er_1w.fillna(0)
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        sc_1w = (er_1w * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_1w = np.zeros(len(close_1w))
        kama_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
        kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
        # 1w trend filter: price above/below 1w KAMA
        trend_filter = close > kama_1w_aligned  # Long bias when above
    else:
        trend_filter = np.ones(n, dtype=bool)  # No filter if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: only trade when market is ranging (chop > 61.8)
        chop_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (momentum fading) or chop regime ends
            if rsi[i] > 50 or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (momentum fading) or chop regime ends
            if rsi[i] < 50 or not chop_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if chop_regime and trend_filter[i]:
                # Long entry: RSI < 30 (oversold) in ranging market
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI > 70 (overbought) in ranging market
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals