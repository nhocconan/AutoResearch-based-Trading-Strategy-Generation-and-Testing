#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX(12) crossover + volume spike + 12h EMA50 trend filter.
Long when TRIX crosses above zero AND volume > 1.5x 20-period average AND close > 12h EMA50.
Short when TRIX crosses below zero AND volume > 1.5x 20-period average AND close < 12h EMA50.
Exit when TRIX crosses zero in opposite direction or ATR stoploss (2.0x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 30-50 trades/year per symbol.
TRIX captures momentum with less whipsaw than MACD. Volume confirmation ensures strong breakouts.
12h EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades. Works in both bull and bear regimes by following momentum.
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
    
    # Load 4h data for price action and ATR - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate TRIX(12) on 4h close: triple EMA of percent change
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then percent change
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Avoid division by zero: add small epsilon to ema3
    trix = 100 * (pd.Series(ema3).pct_change().values)
    trix[0] = 0.0  # first value is NaN from pct_change
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        
        if position == 0:
            # Long: TRIX crosses above zero AND volume spike AND close > 12h EMA50
            if (trix_prev <= 0 and trix_now > 0 and 
                volume[i] > 1.5 * vol_ma_val and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TRIX crosses below zero AND volume spike AND close < 12h EMA50
            elif (trix_prev >= 0 and trix_now < 0 and 
                  volume[i] > 1.5 * vol_ma_val and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: TRIX crosses below zero or ATR stoploss
                if trix_prev >= 0 and trix_now < 0:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: TRIX crosses above zero or ATR stoploss
                if trix_prev <= 0 and trix_now > 0:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_TRIX12_VolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0