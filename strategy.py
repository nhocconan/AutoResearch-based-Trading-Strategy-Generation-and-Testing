#!/usr/bin/env python3
"""
4h_1d_KAMA_ChoppinessRegime_V1
Hypothesis: Use 1d KAMA direction + 4h RSI(14) + 4h Choppiness Index regime filter.
- Only go long when 1d KAMA is rising (bullish bias) and 4h RSI < 30 (oversold) and market is trending (CHOP < 38.2)
- Only go short when 1d KAMA is falling (bearish bias) and 4h RSI > 70 (overbought) and market is trending (CHOP < 38.2)
- In ranging markets (CHOP >= 38.2), stay flat to avoid whipsaw
- Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn
- Works in bull (KAMA up + RSI dips = buy) and bear (KAMA down + RSI spikes = sell) regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d KAMA
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d KAMA (30, 2, 30) ===
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    volatility[0] = direction[0]  # avoid division by zero
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc[0] = (2/(2+1) - 2/(30+1)) + 2/(30+1)  # initial value
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.diff(kama, prepend=0)
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    # Align KAMA direction to 4h
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir.astype(float))
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(atr_period)
    # Handle division by zero or invalid values
    chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_dir_val = kama_dir_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: KAMA up + RSI oversold + trending market
            if kama_dir_val == 1 and rsi_val < 30 and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + trending market
            elif kama_dir_val == -1 and rsi_val > 70 and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA turns down OR RSI overbought OR market becomes ranging
            if kama_dir_val == -1 or rsi_val > 70 or chop_val >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up OR RSI oversold OR market becomes ranging
            if kama_dir_val == 1 or rsi_val < 30 or chop_val >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_ChoppinessRegime_V1"
timeframe = "4h"
leverage = 1.0