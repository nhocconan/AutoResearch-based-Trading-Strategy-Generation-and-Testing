#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX(12) zero-cross with 1d EMA50 trend filter and volume spike confirmation.
Long when TRIX crosses above zero, close > 1d EMA50 (uptrend), and volume > 1.8x average.
Short when TRIX crosses below zero, close < 1d EMA50 (downtrend), and volume > 1.8x average.
Exit on opposite TRIX cross or trend reversal. Uses 12h timeframe targeting 50-150 total trades over 4 years.
TRIX filters noise and identifies momentum shifts, EMA50 provides medium-term trend bias,
volume spike confirms breakout strength. Designed to capture sustained moves in both bull and bear markets
while avoiding whipsaws in choppy conditions through momentum confirmation.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # TRIX calculation on primary timeframe (12h)
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1 period ago
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix_values = trix.values
    trix_prev = trix_values[:-1]  # previous bar TRIX
    trix_cross_up = (trix_values > 0) & (trix_prev <= 0)  # crossed above zero
    trix_cross_down = (trix_values < 0) & (trix_prev >= 0)  # crossed below zero
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(trix_values[i]) or np.isnan(trix_prev[i] if i > 0 else np.nan)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: TRIX crosses above zero AND price > 1d EMA50 (uptrend) AND volume spike
            if (trix_cross_up[i] and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TRIX crosses below zero AND price < 1d EMA50 (downtrend) AND volume spike
            elif (trix_cross_down[i] and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: TRIX crosses below zero OR trend reversal
                if (trix_cross_down[i] or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: TRIX crosses above zero OR trend reversal
                if (trix_cross_up[i] or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_TRIX_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0