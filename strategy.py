#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams %R extremes with 4h EMA50 trend filter and volume confirmation.
Long when Williams %R(14) < -80 (oversold) AND price > 4h EMA50 AND volume > 1.5x 20-period average.
Short when Williams %R(14) > -20 (overbought) AND price < 4h EMA50 AND volume > 1.5x 20-period average.
Exit when Williams %R returns to -50 (mean reversion) or ATR trailing stop hit (2.0*ATR from extreme).
Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in both ranging and trending markets.
The 4h EMA50 ensures we trade with the intermediate trend, reducing counter-trend whipsaws.
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.25) controls risk and fee churn.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_since_entry = 0.0  # highest for long, lowest for short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Williams %R needs 14, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr_val = williams_r_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 4h EMA50 AND volume spike
            if (wr_val < -80.0 and price > ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_since_entry = price  # track highest for long
            # Short: Williams %R overbought (> -20) AND price < 4h EMA50 AND volume spike
            elif (wr_val > -20.0 and price < ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_since_entry = price  # track lowest for short
        else:
            # Update extreme since entry for trailing stop
            if position == 1:
                extreme_since_entry = max(extreme_since_entry, price)
            elif position == -1:
                extreme_since_entry = min(extreme_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R returns to -50 (mean reversion)
            if position == 1 and wr_val >= -50.0:
                exit_signal = True
            elif position == -1 and wr_val <= -50.0:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from extreme since entry
            if position == 1 and price < extreme_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > extreme_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                extreme_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extremes_4hEMA50_Trend_VolumeConfirmation_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0