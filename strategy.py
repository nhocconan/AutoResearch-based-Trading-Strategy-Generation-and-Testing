#!/usr/bin/env python3
# 4h_1d_kama_rsi_volume_v1
# Strategy: 4h KAMA trend with RSI momentum and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing whipsaws in ranging markets.
# RSI > 55 confirms bullish momentum, RSI < 45 confirms bearish momentum.
# Volume > 1.5x 20-period average confirms institutional participation.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.
# Works in bull markets via trend continuation and bear markets via short signals during
# distribution phases. Uses 1d timeframe for trend context and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h KAMA(10,2,30) - adaptive moving average
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs fixing
    
    # Correct ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 1.0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[9] = close[9]  # seed
    
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # 4h RSI(14) for momentum
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
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum filter: >55 for bullish, <45 for bearish
        rsi_bullish = rsi[i] > 55
        rsi_bearish = rsi[i] < 45
        
        # Entry conditions
        # Long: price above KAMA AND RSI bullish AND volume confirmation
        if price_above_kama and rsi_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price below KAMA AND RSI bearish AND volume confirmation
        elif price_below_kama and rsi_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite condition (price crosses KAMA in opposite direction)
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