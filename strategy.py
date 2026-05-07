#!/usr/bin/env python3
name = "6h_Keltner_Channel_Breakout_1dATR_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA(20) and ATR(10) for Keltner Channel on 6h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr10_raw = pd.Series(high - low).rolling(window=10, min_periods=10).mean().values
    
    upper_keltner = ema20 + 2.0 * atr10_raw
    lower_keltner = ema20 - 2.0 * atr10_raw
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr14_1d_raw = np.maximum(high_1d - low_1d,
                              np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                         np.abs(low_1d - np.roll(close_1d, 1))))
    atr14_1d = pd.Series(atr14_1d_raw).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 6h
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Wait for EMA20 and ATR10
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or np.isnan(atr14_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility regimes
        if atr14_1d_aligned[i] < 0.01 * close[i]:  # Skip if ATR too low relative to price
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above upper Keltner with volume filter
            if close[i] > upper_keltner[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Keltner with volume filter
            elif close[i] < lower_keltner[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA(20) or volatility drops
            if close[i] < ema20[i] or atr14_1d_aligned[i] < 0.008 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA(20) or volatility drops
            if close[i] > ema20[i] or atr14_1d_aligned[i] < 0.008 * close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner Channel breakout on 6h with 1d ATR volatility filter and volume confirmation.
# Long when price breaks above upper Keltner (EMA20 + 2*ATR) with volume filter in normal/high volatility.
# Short when price breaks below lower Keltner (EMA20 - 2*ATR) with volume filter.
# Uses volatility filter to avoid whipsaws in low-volatility regimes.
# Designed for 6h timeframe to target 50-150 total trades over 4 years.
# Works in both trending (breakouts) and ranging (mean-reversion via volatility filter) markets.