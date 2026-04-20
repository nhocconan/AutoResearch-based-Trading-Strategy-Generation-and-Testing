# [EXPERIMENT #65468] 12h_KAMA_Direction_With_Volume_and_Chop_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise—slow in ranging, fast in trending.
# Direction: price > KAMA = bullish, price < KAMA = bearish. Entry when price crosses KAMA with volume confirmation.
# Chop filter: Choppiness Index > 61.8 = ranging (avoid trend trades), < 38.2 = trending (allow trades).
# Works in bull/bear via directional entries; volatility-adjusted via Chop regime.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_KAMA_Direction_With_Volume_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # ER = |net change| / sum(|abs change|) over 10 periods
    # SCR = [ER * (fastest - slowest) + slowest]^2
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = change
    
    net_change = np.abs(np.subtract(close, np.roll(close, 10)))
    sum_abs = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 10), axis=1)
    # Handle edge cases for sum_abs calculation
    sum_abs = np.convolve(abs_change, np.ones(10), mode='full')[:n] * 0  # placeholder, will compute properly
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        net = np.abs(close[i] - close[i-10])
        total = np.sum(np.abs(np.diff(close[i-9:i+1])))
        er[i] = net / total if total != 0 else 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Choppiness Index (14-period) ===
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14)
    
    # Chop regimes: >61.8 = ranging, <38.2 = trending
    chop_trending = chop < 38.2  # Only trade in trending markets
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (Chop < 38.2)
        if not chop_trending[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals