#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and ATR volatility regime.
Bull Power = High - EMA13, Bear Power = Low - EMA13. In uptrends (price > 1d EMA50), look for Bull Power expansion (>0 and rising) for longs.
In downtrends (price < 1d EMA50), look for Bear Power contraction (<0 and falling) for shorts. Uses ATR to normalize threshold and avoid whipsaw in ranging markets.
Designed for 6h timeframe to achieve 12-30 trades/year with discrete sizing (0.25) to minimize fee drag. Works in both bull (trend continuation) and bear (mean reversion within trend) markets.
"""

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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 6h ATR(14) for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar TR = high - low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Normalize Elder Ray by ATR
    bull_power_norm = bull_power / atr
    bear_power_norm = bear_power / atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 14)  # need EMA50, EMA13, ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power_norm[i]) or np.isnan(bear_power_norm[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (price > 1d EMA50) AND Bull Power expanding (>0 and rising)
            if close[i] > ema_50_aligned[i] and bull_power_norm[i] > 0 and bull_power_norm[i] > bull_power_norm[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < 1d EMA50) AND Bear Power contracting (<0 and falling)
            elif close[i] < ema_50_aligned[i] and bear_power_norm[i] < 0 and bear_power_norm[i] < bear_power_norm[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Trend reversal OR Elder Ray divergence
            exit_signal = False
            if position == 1:
                # Exit long when trend breaks OR Bull Power deteriorates (<=0 or falling)
                if close[i] <= ema_50_aligned[i] or bull_power_norm[i] <= 0 or bull_power_norm[i] < bull_power_norm[i-1]:
                    exit_signal = True
            elif position == -1:
                # Exit short when trend breaks OR Bear Power expands (>=0 or rising)
                if close[i] >= ema_50_aligned[i] or bear_power_norm[i] >= 0 or bear_power_norm[i] > bear_power_norm[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_Trend_1dEMA50_ATR_Normalized"
timeframe = "6h"
leverage = 1.0