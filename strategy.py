#!/usr/bin/env python3
"""
6h Keltner Channel Breakout + 12h ADX Trend + Volume Confirmation
Long when price breaks above upper KC with ADX>25 and volume spike; short when breaks below lower KC.
Exit when price crosses middle line (EMA20). Uses 12h ADX for trend strength filter.
Designed to work in both bull (breakouts) and bear (mean reversion via exits) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_breakout_12h_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Keltner Channel (20, 2) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_kc = ema20 + 2 * atr
    lower_kc = ema20 - 2 * atr
    
    # === Volume Spike Detector ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # === 12h ADX Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(ema20[i]) or np.isnan(upper_kc[i]) or np.isnan(lower_kc[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA20 (middle line)
            if close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA20 (middle line)
            if close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: break above upper KC + ADX>25 + volume spike
            if (close[i] > upper_kc[i] and 
                adx_aligned[i] > 25 and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: break below lower KC + ADX>25 + volume spike
            elif (close[i] < lower_kc[i] and 
                  adx_aligned[i] > 25 and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals