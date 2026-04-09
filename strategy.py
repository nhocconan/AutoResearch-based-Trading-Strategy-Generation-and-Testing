#!/usr/bin/env python3
# 12h_kama_rsi_chop_v1
# Hypothesis: 12h KAMA trend direction + RSI mean reversion + chop regime filter. KAMA adapts to volatility for trend detection, RSI(14) provides entry timing in pullbacks, chop filter avoids whipsaws in ranging markets. Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe. Works in bull/bear via adaptive trend + mean reversion logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    diff_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    direction_1d = np.abs(close_1d - np.roll(close_1d, 10))
    volatility_1d = pd.Series(diff_1d).rolling(window=10, min_periods=1).sum().values
    er_1d = np.where(volatility_1d != 0, direction_1d / volatility_1d, 0)
    sc_1d = (er_1d * (2/2 - 2/30) + 2/30) ** 2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    
    # Calculate 1d RSI(14)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Calculate 1d Chopiness Index(14)
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / np.log10(14)) / np.log10((hh_14 - ll_14) + 1e-10)
    
    # Align 1d indicators to 12h timeframe (completed daily candle only)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price < KAMA (trend reversal)
            if rsi_1d_aligned[i] > 70 or close[i] < kama_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price > KAMA (trend reversal)
            if rsi_1d_aligned[i] < 30 or close[i] > kama_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price > KAMA (uptrend) + RSI < 40 (pullback) + chop > 61.8 (ranging)
            if (close[i] > kama_1d_aligned[i]) and (rsi_1d_aligned[i] < 40) and (chop_1d_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price < KAMA (downtrend) + RSI > 60 (pullback) + chop > 61.8 (ranging)
            elif (close[i] < kama_1d_aligned[i]) and (rsi_1d_aligned[i] > 60) and (chop_1d_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals