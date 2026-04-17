#!/usr/bin/env python3
"""
12h_RSI_Range_Reversal_v1
RSI-based mean reversion on 12h timeframe with volume confirmation and trend filter.
Uses daily close trend (EMA50) to determine market bias: long only when price above EMA50,
short only when price below EMA50. Enters at RSI extremes (oversold/overbought) with volume spike.
Exits when RSI returns to neutral zone (40-60). Designed to work in both bull and bear markets
by aligning with higher timeframe trend while capturing mean reversion moves.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 with proper initialization
    ema_50 = np.full_like(close_1d, np.nan)
    multiplier = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50[i] = close_1d[i]
        elif i < 50:
            # Simple average of available data
            ema_50[i] = np.mean(close_1d[:i+1])
        else:
            ema_50[i] = (close_1d[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # === 12h RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 20:
            vol_ma_20[i] = np.mean(volume[:i+1]) if i >= 0 else np.nan
        else:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d EMA50 to 12h timeframe ===
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA50 (uptrend bias) AND RSI oversold (<30) AND volume confirmation
            if (close[i] > ema_50_aligned[i] and 
                rsi[i] < 30 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA50 (downtrend bias) AND RSI overbought (>70) AND volume confirmation
            elif (close[i] < ema_50_aligned[i] and 
                  rsi[i] > 70 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI returns to neutral (>=40) OR price crosses below EMA50
            if (rsi[i] >= 40 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<=60) OR price crosses above EMA50
            if (rsi[i] <= 60 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Range_Reversal_v1"
timeframe = "12h"
leverage = 1.0