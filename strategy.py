#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and volume > 1.5x average.
Short when Bear Power < 0, Bull Power > 0, ADX > 25 (trending), and volume > 1.5x average.
Exit when ADX < 20 (range) or power signals reverse. Uses 6h timeframe targeting 50-150 total trades over 4 years.
Elder Ray measures bull/bear strength via EMA13, ADX filters for trending markets only to avoid whipsaws,
volume confirms momentum. Designed to work in both bull (strong uptrends) and bear (strong downtrends) regimes
by only taking trades when trend is strong (ADX > 25).
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
    
    # Load 1d data for EMA13 and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 on 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX on 1d
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = low_1d[0]
    up_move = high_1d - prev_high_1d
    down_move = prev_low_1d - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI, -DI, DX, ADX
    plus_di_1d = 100 * plus_dm_smooth / tr_smooth
    minus_di_1d = 100 * minus_dm_smooth / tr_smooth
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_val = ema13_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Calculate Elder Ray Power
        bull_power = price - ema13_val
        bear_power = ema13_val - price  # inverse of bull power
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0 (redundant but explicit), ADX > 25, volume spike
            if (bull_power > 0 and bear_power < 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0 (redundant but explicit), ADX > 25, volume spike
            elif (bear_power < 0 and bull_power > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (range) OR Bull Power <= 0
                if (adx_val < 20 or bull_power <= 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (range) OR Bear Power >= 0
                if (adx_val < 20 or bear_power >= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0