#!/usr/bin/env python3
# 4h_kama_macd_volatility_regime_v1
# Hypothesis: 4h strategy combining Kaufman Adaptive Moving Average (KAMA) trend direction,
# MACD histogram momentum, and volatility regime filter (ATR ratio) works in both bull and bear markets.
# Long: KAMA upward (price > KAMA) + MACD histogram > 0 + ATR(7)/ATR(30) < 1.2 (low volatility)
# Short: KAMA downward (price < KAMA) + MACD histogram < 0 + ATR(7)/ATR(30) < 1.2 (low volatility)
# Exit: Opposite KAMA cross or volatility expansion (ATR ratio > 1.5)
# Uses 4h primary timeframe with 1h HTF for volatility regime to avoid look-ahead.
# Target: 80-150 total trades over 4 years (20-38/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_macd_volatility_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(30) and ATR(7) for volatility regime with min_periods
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr30 = np.full(n, np.nan)
    atr7 = np.full(n, np.nan)
    for i in range(30, n):
        atr30[i] = np.mean(tr[i-30:i])
    for i in range(7, n):
        atr7[i] = np.mean(tr[i-7:i])
    # ATR ratio: ATR(7)/ATR(30) - low when < 1.0, high when > 1.5
    atr_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr30[i] > 0:
            atr_ratio[i] = atr7[i] / atr30[i]
    
    # Calculate KAMA(10) - smoothing constant based on efficiency ratio
    # ER = |close - close[10]| / sum(|close - close[1]| over 10 periods)
    # SC = [ER * (fastest - slowest) + slowest]^2 where fastest=2/(2+1), slowest=2/(30+1)
    direction = np.abs(np.subtract(close[10:], close[:-10]))  # length n-10
    volatility = np.zeros(n-1)
    for i in range(1, n):
        volatility[i] = abs(close[i] - close[i-1])
    # Sum volatility over 10 periods
    vol_sum = np.full(n, np.nan)
    for i in range(10, n):
        vol_sum[i] = np.sum(volatility[i-9:i+1])  # 10 periods
    er = np.full(n, np.nan)
    for i in range(10, n):
        if vol_sum[i] > 0:
            er[i] = direction[i-10] / vol_sum[i]
        else:
            er[i] = 0
    fastest = 2.0 / (2 + 1)   # 0.6667
    slowest = 2.0 / (30 + 1)  # 0.0645
    sc = np.square(er * (fastest - slowest) + slowest)
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate MACD(12,26,9)
    # EMA12 = close.ewm(span=12, adjust=False).mean()
    # EMA26 = close.ewm(span=26, adjust=False).mean()
    # MACD = EMA12 - EMA26
    # Signal = MACD.ewm(span=9, adjust=False).mean()
    # Histogram = MACD - Signal
    ema12 = np.full(n, np.nan)
    ema26 = np.full(n, np.nan)
    macd = np.full(n, np.nan)
    signal_line = np.full(n, np.nan)
    histogram = np.full(n, np.nan)
    
    # Calculate EMAs
    multiplier_12 = 2.0 / (12 + 1)
    multiplier_26 = 2.0 / (26 + 1)
    multiplier_9 = 2.0 / (9 + 1)
    
    ema12[11] = np.mean(close[:12])  # seed
    ema26[25] = np.mean(close[:26])  # seed
    for i in range(12, n):
        ema12[i] = close[i] * multiplier_12 + ema12[i-1] * (1 - multiplier_12)
    for i in range(26, n):
        ema26[i] = close[i] * multiplier_26 + ema26[i-1] * (1 - multiplier_26)
    
    # Calculate MACD and signal
    for i in range(26, n):
        macd[i] = ema12[i] - ema26[i]
    for i in range(35, n):  # 26+9=35
        signal_line[i] = macd[i] * multiplier_9 + signal_line[i-1] * (1 - multiplier_9)
    for i in range(35, n):
        histogram[i] = macd[i] - signal_line[i]
    
    # Get 1h data for volatility regime confirmation (to avoid look-ahead)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate ATR(14) on 1h
    tr_1h = np.zeros(len(df_1h))
    for i in range(1, len(df_1h)):
        tr_1h[i] = max(high_1h[i] - low_1h[i], abs(high_1h[i] - close_1h[i-1]), abs(low_1h[i] - close_1h[i-1]))
    atr_1h = np.full(len(df_1h), np.nan)
    for i in range(14, len(df_1h)):
        atr_1h[i] = np.mean(tr_1h[i-14:i])
    
    # Calculate 1h ATR ratio (ATR7/ATR30) for regime
    atr7_1h = np.full(len(df_1h), np.nan)
    atr30_1h = np.full(len(df_1h), np.nan)
    for i in range(7, len(df_1h)):
        atr7_1h[i] = np.mean(tr_1h[i-7:i])
    for i in range(30, len(df_1h)):
        atr30_1h[i] = np.mean(tr_1h[i-30:i])
    atr_ratio_1h = np.full(len(df_1h), np.nan)
    for i in range(30, len(df_1h)):
        if atr30_1h[i] > 0:
            atr_ratio_1h[i] = atr7_1h[i] / atr30_1h[i]
    
    # Align 1h ATR ratio to 4h timeframe
    atr_ratio_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_ratio_1h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(histogram[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(atr_ratio_1h_aligned[i])):
            # Hold current position if any, otherwise flat
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        vol_regime = atr_ratio_1h_aligned[i] < 1.2  # Low volatility regime
        vol_expansion = atr_ratio[i] > 1.5  # Volatility expansion for exit
        
        if position == 1:  # Long position
            # Exit: KAMA cross down OR volatility expansion
            if close[i] <= kama[i] or vol_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA cross up OR volatility expansion
            if close[i] >= kama[i] or vol_expansion:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price > KAMA + MACD histogram > 0 + low volatility regime
            if (close[i] > kama[i] and histogram[i] > 0 and vol_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: price < KAMA + MACD histogram < 0 + low volatility regime
            elif (close[i] < kama[i] and histogram[i] < 0 and vol_regime):
                position = -1
                signals[i] = -0.25
    
    return signals