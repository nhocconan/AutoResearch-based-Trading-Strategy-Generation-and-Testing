#!/usr/bin/env python3
"""
12h_TRIX_Volume_Spike_Regime_v1
TRIX(15) zero-cross with volume spike and ADX regime filter for 12h timeframe.
Long when TRIX crosses above zero with volume > 1.5x average and ADX > 20.
Short when TRIX crosses below zero with volume > 1.5x average and ADX > 20.
Exit when TRIX reverses direction.
Designed to capture momentum shifts with volume confirmation in trending markets.
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
    volume = prices['volume'].values
    
    # === TRIX(15) calculation ===
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Calculate percent rate of change of triple EMA
    trix = 100 * (np.diff(ema3, prepend=ema3[0]) / (ema3 + 1e-10))
    
    # === Volume average for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ADX(14) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === 1d EMA50 for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # TRIX zero-cross detection
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: TRIX crosses above zero, volume confirmed, ADX > 20, price above 1d EMA50
            if (trix_cross_up and 
                vol_confirmed and 
                adx[i] > 20 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: TRIX crosses below zero, volume confirmed, ADX > 20, price below 1d EMA50
            elif (trix_cross_down and 
                  vol_confirmed and 
                  adx[i] > 20 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: TRIX reverses direction
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_Volume_Spike_Regime_v1"
timeframe = "12h"
leverage = 1.0