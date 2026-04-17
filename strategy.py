#!/usr/bin/env python3
"""
12h_TripleConfirmation_Strategy_v1
Triple confirmation strategy on 12h: 
1. Trend: 12h price above/below 1d SMA50 (trend filter)
2. Momentum: 12h RSI(14) crossing 50 (momentum entry)
3. Volume: 12h volume > 1.5x 20-period average (confirmation)
Designed for low-frequency, high-conviction trades in both bull and bear markets.
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
    
    # === 1d SMA50 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate SMA50 with min_periods
    sma50_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 49:  # 50 periods
            sma50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    # === 12h RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:  # 20 periods
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d SMA50 to 12h timeframe ===
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above 1d SMA50 (uptrend) AND RSI crosses above 50 AND volume confirmation
            if (close[i] > sma50_1d_aligned[i] and 
                rsi[i] > 50 and 
                (i == warmup or rsi[i-1] <= 50) and  # RSI crossing above 50
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below 1d SMA50 (downtrend) AND RSI crosses below 50 AND volume confirmation
            elif (close[i] < sma50_1d_aligned[i] and 
                  rsi[i] < 50 and 
                  (i == warmup or rsi[i-1] >= 50) and  # RSI crossing below 50
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses below 40 OR price crosses below 1d SMA50
            if (rsi[i] < 40 and 
                (i == warmup or rsi[i-1] >= 40)) or \
               close[i] < sma50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 60 OR price crosses above 1d SMA50
            if (rsi[i] > 60 and 
                (i == warmup or rsi[i-1] <= 60)) or \
               close[i] > sma50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TripleConfirmation_Strategy_v1"
timeframe = "12h"
leverage = 1.0