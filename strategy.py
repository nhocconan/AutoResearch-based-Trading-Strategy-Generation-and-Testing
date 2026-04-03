#!/usr/bin/env python3
"""
Experiment #179: 6h Elder Ray + Regime Filter

HYPOTHESIS: Elder Ray (Bull Power/Bear Power) combined with ADX regime filter captures 
momentum exhaustion points. Long when Bull Power > 0 and ADX < 25 (range), 
Short when Bear Power < 0 and ADX < 25. Uses 1d EMA200 as trend filter to avoid 
counter-trend trades. Works in both bull/bear markets by adapting to regimes.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_to_ltf, get_htf_data

name = "mtf_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 6h Indicators ===
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13  
    bear_power = low - ema_13
    
    # ADX(14) for regime detection
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed TR, +DM, -DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    # DI+
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # ADX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for EMA200 and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime and Trend Filters ---
        # Range regime: ADX < 25
        in_range = adx[i] < 25
        # Trend filter: price above/below 1d EMA200
        above_ema200 = close[i] > ema_200_1d_aligned[i]
        below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        # --- Position Management ---
        if in_position:
            # Stoploss: 3 * ATR(14) approximation using 20-period high-low range
            # Simplified: exit if price moves against position by 3% of ATR equivalent
            if position_side > 0:  # Long
                if close[i] < entry_price * 0.97:  # ~3% stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price * 1.03:  # ~3% stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bull Power > 0 (buying pressure) + in range + above EMA200
        if bull_power[i] > 0 and in_range and above_ema200:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Bear Power < 0 (selling pressure) + in range + below EMA200
        elif bear_power[i] < 0 and in_range and below_ema200:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals