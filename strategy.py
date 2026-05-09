#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend (ATR=10, mult=3) with 1d volume confirmation and 12h trend filter.
# Long when price > Supertrend line + volume > 1.5x 20-period EMA + 12h EMA50 up.
# Short when price < Supertrend line + volume > 1.5x 20-period EMA + 12h EMA50 down.
# Uses dynamic stop via Supertrend reversal (no separate stop needed).
# Designed to capture trends while filtering noise with volume and higher timeframe trend.
# Works in both bull and bear markets by following 12h EMA50 direction.
name = "6h_Supertrend_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def supertrend(high, low, close, atr_length=10, multiplier=3):
    """Calculate Supertrend indicator"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR
    atr = np.zeros_like(close)
    atr[:atr_length-1] = np.nan
    atr[atr_length-1] = np.mean(tr[:atr_length])
    for i in range(atr_length, len(close)):
        atr[i] = (atr[i-1] * (atr_length-1) + tr[i]) / atr_length
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    final_ub[:] = np.nan
    final_lb[:] = np.nan
    
    for i in range(atr_length, len(close)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close)
    supertrend[:] = np.nan
    
    for i in range(atr_length, len(close)):
        if i == atr_length:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    return supertrend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Supertrend (10, 3)
    st = supertrend(high, low, close, 10, 3)
    
    # 12h EMA50 trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(st[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > Supertrend + volume confirmation + 12h EMA50 up
            if price > st[i] and vol_confirm[i] and price > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Supertrend + volume confirmation + 12h EMA50 down
            elif price < st[i] and vol_confirm[i] and price < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Supertrend (trend change)
            if price < st[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Supertrend (trend change)
            if price > st[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals