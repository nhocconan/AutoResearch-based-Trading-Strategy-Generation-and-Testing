#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume spike confirmation.
Long when Bull Power > 0 AND ADX > 25 (trending) AND volume > 1.5x average.
Short when Bear Power < 0 AND ADX > 25 (trending) AND volume > 1.5x average.
Elder Ray measures bull/bear strength relative to EMA13; ADX filters for trending markets only.
Designed to capture strong trends in both bull and bear markets while avoiding chop.
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: high - EMA13
    bear_power = low - ema13   # Bear Power: low - EMA13
    
    # Load 12h data for ADX regime filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    c_12h_prev = np.roll(close_12h, 1)
    c_12h_prev[0] = np.nan
    tr = true_range(high_12h, low_12h, c_12h_prev)
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=np.nan)
    down_move = -np.diff(low_12h, prepend=np.nan)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        adx_val = adx_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND ADX > 25 (trending) AND volume confirmation
            if (bull_val > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND ADX > 25 (trending) AND volume confirmation
            elif (bear_val < 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray power reverses OR ADX weakens (chop) OR volume drops
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 OR ADX < 20 (chop) OR volume < average
                if (bull_val <= 0 or adx_val < 20 or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power >= 0 OR ADX < 20 (chop) OR volume < average
                if (bear_val >= 0 or adx_val < 20 or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_12hADX_Volume"
timeframe = "6h"
leverage = 1.0