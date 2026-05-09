#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 12-hour Supertrend (ATR=10, mult=3) for trend direction and 12-hour RSI(14) for mean-reversion entries.
# In uptrend (Supertrend up), go long when RSI crosses below 30 (oversold pullback).
# In downtrend (Supertrend down), go short when RSI crosses above 70 (overbought bounce).
# Exit when RSI returns to neutral zone (40-60) or trend reverses.
# This combines trend following with mean-reversion entries to work in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_Supertrend_RSI_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 12-hour Supertrend for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # True Range
    prev_close = np.roll(df_12h['close'], 1)
    prev_close[0] = np.nan
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - prev_close)
    tr3 = np.abs(df_12h['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (df_12h['high'] + df_12h['low']) / 2.0
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Supertrend calculation
    supertrend = np.zeros_like(df_12h['close'])
    direction = np.ones_like(df_12h['close'])  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_12h)):
        if df_12h['close'].iloc[i] > upper_band[i-1]:
            direction[i] = 1
        elif df_12h['close'].iloc[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Supertrend trend direction (1 = uptrend, -1 = downtrend)
    trend_direction = direction
    trend_direction_aligned = align_htf_to_ltf(prices, df_12h, trend_direction)
    
    # Calculate 12-hour RSI(14)
    delta = pd.Series(df_12h['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_direction_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = trend_direction_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: uptrend + RSI oversold (< 30)
            if trend == 1 and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + RSI overbought (> 70)
            elif trend == -1 and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (> 40) or trend reverses to downtrend
            if rsi_val > 40 or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (< 60) or trend reverses to uptrend
            if rsi_val < 60 or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals