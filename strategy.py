#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 20-day high AND close > 1w EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below 20-day low AND close < 1w EMA34 AND volume > 2.0x 20-period average.
Exit when price retraces to 10-day EMA or ATR trailing stop (1.5*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
Donchian breakouts capture strong momentum moves, and the 1w EMA34 filter ensures alignment with
longer-term trend, reducing whipsaws in both bull and bear markets. Volume confirmation adds
institutional participation validation. Target trade frequency: 7-25 trades/year per symbol.
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d Donchian channels (20-period)
    if len(close) < 20:
        return np.zeros(n)
    
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 10, 14, 20)  # EMA34 needs 34, Donchian needs 20, etc.
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema10[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1w_aligned[i]
        highest_20_val = highest_20[i]
        lowest_20_val = lowest_20[i]
        ema10_val = ema10[i]
        
        if position == 0:
            # Long: Break above 20-day high AND uptrend (price > 1w EMA34) AND volume spike
            if close[i] > highest_20_val and close[i] > ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below 20-day low AND downtrend (price < 1w EMA34) AND volume spike
            elif close[i] < lowest_20_val and close[i] < ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 10-day EMA
            if position == 1 and close[i] <= ema10_val:
                exit_signal = True
            elif position == -1 and close[i] >= ema10_val:
                exit_signal = True
            
            # ATR-based trailing stop: 1.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirmation_EMA10Exit_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0