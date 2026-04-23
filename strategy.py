#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w EMA crossover with volume confirmation and ATR trailing stop.
Long when 1w EMA21 crosses above EMA50 AND daily volume > 1.5x 20-day average.
Short when 1w EMA21 crosses below EMA50 AND daily volume > 1.5x 20-day average.
Exit when price retraces to the 1w EMA34 or ATR trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 1d timeframe to target 15-30 trades/year per symbol (60-120 total over 4 years).
Weekly EMA crossover captures medium-term trend shifts, volume confirmation avoids false signals,
and ATR stop manages risk. Works in both bull and bear markets by following the weekly trend.
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
    
    # Calculate 1w EMA indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA21, EMA50, EMA34
    close_1w = pd.Series(df_1w['close'].values)
    ema21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily volume average (20-period)
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
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # weekly EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema21_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema21_val = ema21_aligned[i]
        ema50_val = ema50_aligned[i]
        ema34_val = ema34_aligned[i]
        
        if position == 0:
            # Long: Weekly EMA21 crosses above EMA50 AND volume spike
            if (ema21_val > ema50_val and ema21_aligned[i-1] <= ema50_aligned[i-1] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Weekly EMA21 crosses below EMA50 AND volume spike
            elif (ema21_val < ema50_val and ema21_aligned[i-1] >= ema50_aligned[i-1] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1w EMA34
            if position == 1 and price <= ema34_val:
                exit_signal = True
            elif position == -1 and price >= ema34_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyEMA_Crossover_VolumeConfirmation_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0