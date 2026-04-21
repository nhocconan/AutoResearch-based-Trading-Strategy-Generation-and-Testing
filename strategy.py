#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI2_Confirm_v3
Hypothesis: Use KAMA (adaptive moving average) on 1d to determine trend direction, with RSI(2) pullback entries and volume confirmation. 
Designed to capture trend continuation after short-term pullbacks in both bull and bear markets, avoiding whipsaw by using adaptive trend filter.
Target ~15-25 trades/year on 1d by requiring KAMA trend alignment + RSI(2) < 30 for long / > 70 for short + volume > 1.5x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w trend filter: KAMA (adaptive moving average) ===
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, k=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # === RSI(2) on 1d for entry timing ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[1] = np.mean(gain[2:3])  # seed
    avg_loss[1] = np.mean(loss[2:3])
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (2-1) + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * (2-1) + loss[i]) / 2
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    
    # === Volume confirmation: 20-period volume average on 1d ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_ratio = np.where(vol_ma_20_aligned != 0, volume_1d / vol_ma_20_aligned, 1.0)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama_1w_aligned[i]) or
            np.isnan(rsi_2_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_level = kama_1w_aligned[i]
        rsi = rsi_2_aligned[i]
        vol_spike = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI(2) oversold + volume spike
            if (price_close > kama_level and 
                rsi < 30 and 
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI(2) overbought + volume spike
            elif (price_close < kama_level and 
                  rsi > 70 and 
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses KAMA in opposite direction
            if position == 1 and price_close < kama_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > kama_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI2_Confirm_v3"
timeframe = "1d"
leverage = 1.0