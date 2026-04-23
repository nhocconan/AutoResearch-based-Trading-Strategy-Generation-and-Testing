#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX(12) zero-line crossover with 1d EMA34 trend filter and volume confirmation.
Long when TRIX crosses above zero AND 1d EMA34 rising AND volume > 1.8x 20-period MA.
Short when TRIX crosses below zero AND 1d EMA34 falling AND volume > 1.8x 20-period MA.
Exit when TRIX reverses or 1d EMA34 trend changes.
TRIX (Triple Exponential Average) filters noise and identifies momentum shifts effectively in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Calculate TRIX: EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change()).values  # Percentage change of triple EMA
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(36, 34, 20)  # TRIX needs ~36 (3*12), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate TRIX zero-line crossover
        if i >= start_idx + 1:
            trix_prev = trix[i-1]
            trix_cross_up = trix[i] > 0 and trix_prev <= 0
            trix_cross_down = trix[i] < 0 and trix_prev >= 0
        else:
            trix_cross_up = False
            trix_cross_down = False
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_34_aligned[i] > ema_prev
            ema_falling = ema_34_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.8x 20-period MA (higher threshold to reduce trades)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: TRIX crosses above zero AND EMA34 rising AND volume filter
            if trix_cross_up and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND EMA34 falling AND volume filter
            elif trix_cross_down and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: TRIX turns down OR EMA34 starts falling
                if (i >= start_idx + 1 and trix[i] < trix[i-1]) or ema_falling:
                    exit_signal = True
            elif position == -1:
                # Short exit: TRIX turns up OR EMA34 starts rising
                if (i >= start_idx + 1 and trix[i] > trix[i-1]) or ema_rising:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_TRIX_ZeroCross_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0