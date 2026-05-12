#!/usr/bin/env python3
name = "4h_KAMA_Trend_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA parameters (4h) ===
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_length))
    change[0:er_length] = 0
    volatility = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = pd.Series(volatility).rolling(window=er_length, min_periods=er_length).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d data for higher timeframe context ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14) for trend filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Choppiness Index (1d) for regime filter ===
    period = 14
    atr_1d = np.zeros_like(close_1d)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    atr_sum = pd.Series(tr_1d).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (trend up) + RSI > 50 (bullish momentum) + chop < 61.8 (trending market)
            if (close[i] > kama[i] and 
                rsi_1d_aligned[i] > 50 and
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (trend down) + RSI < 50 (bearish momentum) + chop < 61.8 (trending market)
            elif (close[i] < kama[i] and 
                  rsi_1d_aligned[i] < 50 and
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI turns bearish or chop indicates ranging
            if (close[i] < kama[i] or 
                rsi_1d_aligned[i] < 50 or
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI turns bullish or chop indicates ranging
            if (close[i] > kama[i] or 
                rsi_1d_aligned[i] > 50 or
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals