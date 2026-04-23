#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high AND close > 12h EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below 20-period low AND close < 12h EMA50 AND volume > 2.0x 20-period average.
Exit when price retraces to 50% of the breakout range or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag. Targets 12-37 trades/year per symbol.
Donchian channels provide clear breakout levels; EMA50 filters trend direction; volume confirms breakout strength.
Designed to work in both bull (breakouts with volume) and bear (breakdowns with volume) markets.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
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
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    breakout_high = 0.0        # for 50% retracement exit
    breakout_low = 0.0         # for 50% retracement exit
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 2)  # EMA50 needs 50, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_12h_aligned[i]
        high_20 = highest_20[i]
        low_20 = lowest_20[i]
        
        if position == 0:
            # Long: Break above 20-period high AND uptrend (close > EMA50) AND volume spike
            if close[i] > high_20 and close[i] > ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
                breakout_high = high_20  # store breakout level for 50% retracement
            # Short: Break below 20-period low AND downtrend (close < EMA50) AND volume spike
            elif close[i] < low_20 and close[i] < ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
                breakout_low = low_20  # store breakout level for 50% retracement
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 50% of breakout range
            if position == 1:
                retracement_level = breakout_low + 0.5 * (breakout_high - breakout_low)
                if price <= retracement_level:
                    exit_signal = True
            elif position == -1:
                retracement_level = breakout_high - 0.5 * (breakout_high - breakout_low)
                if price >= retracement_level:
                    exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                breakout_high = 0.0
                breakout_low = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirmation_50RetracementExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0