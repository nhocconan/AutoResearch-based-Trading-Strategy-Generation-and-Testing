#!/usr/bin/env python3
"""
4h_TRIX_DMI_Crossover_Volume_Filter
Hypothesis: TRIX (12-period EMA rate-of-change) crossing above/below zero indicates momentum shifts. Combined with DMI (ADX>25) for trend strength and volume > 1.5x 20-period average for confirmation. Designed to capture sustained moves in both bull and bear markets with low trade frequency.
"""
name = "4h_TRIX_DMI_Crossover_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX: 12-period EMA of EMA of EMA of close, then 1-period percent change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = ema3.pct_change() * 100  # Convert to percentage
    
    # DMI (ADX) calculation
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=tr_period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=tr_period, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix[i]) or np.isnan(adx[i]) or 
            np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + ADX > 25 + volume filter
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                adx[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + ADX > 25 + volume filter
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  adx[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX crosses back through zero (opposite direction)
            if position == 1:
                if trix[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals