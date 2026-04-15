#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA Direction with RSI Filter and Volume Confirmation
# Uses KAMA to determine trend direction, RSI for overbought/oversold conditions, and volume for confirmation.
# Works in both bull and bear markets by adapting to volatility via KAMA's efficiency ratio.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for RSI filter (trend context)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate KAMA (2-period ER, 30-period smoothing) on close
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if i == 0:
            er[i] = 0
        else:
            direction = np.abs(close[i] - close[i-14]) if i >= 14 else np.abs(close[i] - close[0])
            volatility_sum = np.sum(np.abs(np.diff(close[max(0, i-13):i+1])))
            er[i] = direction / (volatility_sum + 1e-10) if volatility_sum > 0 else 0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = 50  # initialize
    
    # Align KAMA and RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, close, kama)  # using close as dummy for alignment
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]):
            continue
        
        # Long: price above KAMA, RSI < 70 (not overbought), volume confirmation
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] < 70 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price below KAMA, RSI > 30 (not oversold), volume confirmation
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] > 30 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or RSI extreme
        elif position == 1 and (close[i] < kama_aligned[i] or rsi_aligned[i] > 80):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_aligned[i] or rsi_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Direction_RSI_Volume"
timeframe = "4h"
leverage = 1.0