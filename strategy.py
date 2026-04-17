#!/usr/bin/env python3
"""
4h_ADX_EMA_Crossover_v1
Trend-following strategy using ADX(14) > 25 for trend strength and EMA(21) crossovers for entry.
Exit when ADX drops below 20 or EMA crossover reverses.
Designed to capture strong trends while avoiding choppy markets. Uses 1-day EMA50 as higher timeframe filter.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === EMA21 for entry signal ===
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === EMA8 for entry signal (faster) ===
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # === ADX(14) for trend strength ===
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
    
    # === Volume average for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA50 for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8[i]) or 
            np.isnan(ema21[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: EMA8 crosses above EMA21, ADX > 25, volume confirmed, price above 1d EMA50
            if (ema8[i] > ema21[i] and 
                ema8[i-1] <= ema21[i-1] and 
                adx[i] > 25 and 
                vol_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: EMA8 crosses below EMA21, ADX > 25, volume confirmed, price below 1d EMA50
            elif (ema8[i] < ema21[i] and 
                  ema8[i-1] >= ema21[i-1] and 
                  adx[i] > 25 and 
                  vol_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: trend weakening or EMA crossover reverses
        elif position == 1:
            # Exit long: ADX < 20 OR EMA8 crosses below EMA21
            if (adx[i] < 20 or 
                (ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 OR EMA8 crosses above EMA21
            if (adx[i] < 20 or 
                (ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_EMA_Crossover_v1"
timeframe = "4h"
leverage = 1.0