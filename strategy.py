#!/usr/bin/env python3
# 12h_1d_kama_rsi_v1
# Strategy: 12h KAMA trend with RSI filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI > 50 filters for bullish momentum, RSI < 50 for bearish. Volume > 1.5x 20-period average
# confirms institutional participation. Designed for low trade frequency (~15-30/year) to minimize
# fee drag. Works in bull markets via trend continuation and bear markets via short signals
# during distribution phases. KAMA's adaptive nature reduces whipsaws in sideways markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h KAMA(10,2,30) - adaptive moving average
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 12h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum filter: >50 for bullish, <50 for bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Entry conditions
        # Long: price above KAMA AND RSI bullish AND volume confirmation
        if price_above_kama and rsi_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price below KAMA AND RSI bearish AND volume confirmation
        elif price_below_kama and rsi_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses KAMA in opposite direction
        elif position == 1 and price_below_kama:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_above_kama:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals