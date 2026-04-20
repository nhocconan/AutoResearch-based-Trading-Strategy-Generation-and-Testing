#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA + RSI(14) + Chop regime filter (CHOP > 61.8) for mean reversion
# - KAMA identifies trend direction on 12h
# - RSI(14) for overbought/oversold conditions
# - Chop > 61.8 indicates ranging market (good for mean reversion)
# - Long when KAMA up, RSI < 30, Chop > 61.8
# - Short when KAMA down, RSI > 70, Chop > 61.8
# - Exit when RSI crosses 50 or Chop < 38.2 (trending)
# - Uses 1d for Chop calculation, 12h for KAMA/RSI
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for Chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop: (sum(ATR,14) / (max(high,14) - min(low,14))) * 100
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = (atr_sum / range_14) * 100
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load 12h data for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate KAMA (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0)  # This needs correction
    # Recalculate volatility properly
    volatility = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility[i] = volatility[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    # Actually, let's use a simpler ER calculation
    price_change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_sum = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        volatility_sum[i] = volatility_sum[i-1] + np.abs(close_12h[i] - close_12h[i-1])
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h = align_htf_to_ltf(prices, df_12h, rsi)
    
    # 12h price data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(chop_12h[i]) or np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_12h[i]
        rsi_val = rsi_12h[i]
        chop_val = chop_12h[i]
        
        if position == 0:
            # Long: KAMA up, RSI oversold, choppy market
            if price > kama_val and rsi_val < 30 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, choppy market
            elif price < kama_val and rsi_val > 70 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses 50 or chop < 38.2 (trending)
            if rsi_val > 50 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses 50 or chop < 38.2 (trending)
            if rsi_val < 50 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_MeanReversion"
timeframe = "12h"
leverage = 1.0