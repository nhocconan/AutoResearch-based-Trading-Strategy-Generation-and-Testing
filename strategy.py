#!/usr/bin/env python3
# 4h_1d_kama_volume_crossover_v1
# Strategy: 4h KAMA direction with volume confirmation and 1d EMA trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# Combined with 1d EMA for trend direction and volume confirmation to filter low-quality breakouts.
# Works in bull markets via long signals and bear markets via short signals.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_volume_crossover_v1"
timeframe = "4h"
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
    
    # 4h KAMA (10-period ER, 2/30 smoothing constants)
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    # Correct ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i == 10:
            change_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
        else:
            change_sum = change_sum - np.abs(close[i-10] - close[i-9]) + np.abs(close[i] - close[i-1])
            volatility_sum = volatility_sum - np.abs(close[i-10] - close[i-9]) + np.abs(close[i] - close[i-1])
        if volatility_sum > 0:
            er[i] = change_sum / volatility_sum
        else:
            er[i] = 0
    # Smoothing constant
    sc = (er * (2/30 - 2/10) + 2/10) ** 2
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # KAMA direction: price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price above KAMA AND bullish trend AND volume confirmation
        if price_above_kama and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price below KAMA AND bearish trend AND volume confirmation
        elif price_below_kama and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite KAMA signal (price below KAMA for long, above for short)
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