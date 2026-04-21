#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with choppiness index regime filter to avoid false signals in sideways markets.
Enter long when price > KAMA and CHOP > 61.8 (ranging market, mean reversion to upside).
Enter short when price < KAMA and CHOP > 61.8 (ranging market, mean reversion to downside).
Exit when price crosses Donchian(20) channel opposite to position.
Uses 1w HTF trend filter (price > 1w EMA200 for longs, price < 1w EMA200 for shorts) to align with major trend.
Discrete sizing: 0.25. Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop: 1w for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Primary 1d indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (10-period ER, 2/30 smoothing constants) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    # Recompute volatility properly
    volatility = np.zeros_like(change)
    for i in range(len(volatility)):
        volatility[i] = np.sum(np.abs(np.diff(close[i:i+10])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- Choppiness Index (14-period) ---
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    
    # --- Donchian Channel (20-period) for exit ---
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w HTF trend filter: EMA200 ===
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        chop_val = chop[i]
        ema_200 = ema_200_1w_aligned[i]
        
        if position == 0:
            # Enter only in choppy market (regime filter) with KAMA alignment
            if chop_val > 61.8:  # ranging market
                if price > kama_val and price > ema_200:  # long bias in uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < kama_val and price < ema_200:  # short bias in downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Exit when price crosses Donchian low (opposite channel)
            if price < dc_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses Donchian high (opposite channel)
            if price > dc_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0