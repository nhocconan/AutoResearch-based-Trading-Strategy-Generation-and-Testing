#!/usr/bin/env python3
# 6H_Daily_Keltner_Breakout_Trend_TrendFilter
# Hypothesis: 6h Keltner breakout with 1d trend filter (EMA50) and volume confirmation.
# Keltner channels (ATR-based) adapt to volatility, reducing false breakouts in low-volatility periods.
# Trend filter ensures we only trade in the direction of the 1d trend, improving win rate in both bull and bear markets.
# Volume confirmation adds conviction to breakouts. Target: 15-35 trades/year.

name = "6H_Daily_Keltner_Breakout_Trend_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and ATR for Keltner channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR(10) for Keltner channels (using 1d data)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First TR is NaN
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels on 1d: EMA20 ± 2*ATR(10)
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_1d + 2 * atr10
    lower_keltner = ema20_1d - 2 * atr10
    
    # Align 1d indicators to 6t
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Keltner + above 1d EMA50 + volume confirmation
            if (close[i] > upper_keltner_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Keltner + below 1d EMA50 + volume confirmation
            elif (close[i] < lower_keltner_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below lower Keltner (reversal signal)
            if close[i] < lower_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above upper Keltner (reversal signal)
            if close[i] > upper_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals