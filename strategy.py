#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop Filter
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI for momentum,
# and Choppiness Index to filter ranging markets. Goes long when KAMA up, RSI > 50, and CHOP < 38.2 (trending).
# Goes short when KAMA down, RSI < 50, and CHOP < 38.2. Works in trending markets (bull/bear) and avoids ranging.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility properly
    volatility = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
    er = np.zeros_like(close)
    er[er_period:] = change[er_period:] / (volatility[er_period:] + 1e-10)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before warmup
    
    # Choppiness Index (14-period)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True range sum over period
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/min close over period
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (max_close - min_close + 1e-10)) / np.log10(14)
    chop[:14] = 50  # neutral before warmup
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(14, n):
        # Long: KAMA up, RSI > 50, CHOP < 38.2 (trending)
        if (kama[i] > kama[i-1] and
            rsi[i] > 50 and
            chop[i] < 38.2 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: KAMA down, RSI < 50, CHOP < 38.2 (trending)
        elif (kama[i] < kama[i-1] and
              rsi[i] < 50 and
              chop[i] < 38.2 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite condition or CHOP > 61.8 (ranging)
        elif position == 1 and (kama[i] < kama[i-1] or rsi[i] < 50 or chop[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (kama[i] > kama[i-1] or rsi[i] > 50 or chop[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0