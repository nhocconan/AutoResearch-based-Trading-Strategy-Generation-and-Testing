#!/usr/bin/env python3
# 1d_1w_volatility_adjusted_keltner_breakout
# Hypothesis: Daily Keltner breakout with weekly volatility filter to capture trend moves in both bull and bear markets.
# Uses weekly ATR to normalize breakout thresholds, reducing false signals during low volatility.
# Targets 10-25 trades/year (40-100 total) to minimize fee drag while maintaining edge.

name = "1d_1w_volatility_adjusted_keltner_breakout"
timeframe = "1d"
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
    
    # Get weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR (14-period) for volatility normalization
    tr1 = np.abs(np.subtract(high_1w, low_1w))
    tr2 = np.abs(np.subtract(high_1w, np.roll(close_1w, 1)))
    tr3 = np.abs(np.subtract(low_1w, np.roll(close_1w, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Keltner Channel (20-period EMA, 2.0 * ATR)
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    atr_10 = pd.Series(np.abs(np.subtract(high, low))).rolling(window=10, min_periods=10).mean().values
    upper_keltner = ema_20 + 2.0 * atr_10
    lower_keltner = ema_20 - 2.0 * atr_10
    
    # Align weekly ATR to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Dynamic breakout threshold based on weekly volatility
        # Higher volatility = wider breakout threshold
        volatility_multiplier = np.clip(atr_1w_aligned[i] / np.nanmedian(atr_1w_aligned), 0.5, 2.0)
        upper_break = ema_20[i] + 2.0 * atr_10[i] * volatility_multiplier
        lower_break = ema_20[i] - 2.0 * atr_10[i] * volatility_multiplier
        
        # Long entry: close breaks above dynamic upper band with volume
        if (close[i] > upper_break and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below dynamic lower band with volume
        elif (close[i] < lower_break and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: close crosses back to EMA(20)
        elif position == 1 and close[i] < ema_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals