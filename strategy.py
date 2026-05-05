#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(12) zero-line crossover + 1d volume spike + ADX regime filter
# TRIX(12) > 0 indicates bullish momentum, < 0 bearish momentum on 4h
# 1d volume > 2.0x 20-period MA confirms institutional participation
# 1d ADX > 20 ensures we trade in trending markets only (avoids chop)
# Long: TRIX crosses above zero AND volume spike AND ADX > 20
# Short: TRIX crosses below zero AND volume spike AND ADX > 20
# Exit: TRIX crosses zero in opposite direction OR volume drops below 1.2x MA OR ADX < 15
# Uses TRIX for smooth momentum, volume for conviction, ADX for regime filter
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_TRIX_1dVolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(12) on 4h close
    if len(close) >= 12:
        # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then percent change
        ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
        ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
        ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
        trix_raw = np.diff(ema3, prepend=ema3[0])
        trix = trix_raw / np.where(np.abs(ema3) > 1e-10, np.abs(ema3), 1e-10) * 100
    else:
        trix = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = vol_1d > (2.0 * vol_ma_20)
        volume_decay = vol_1d < (1.2 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(vol_1d), dtype=bool)
        volume_decay = np.ones(len(vol_1d), dtype=bool)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / np.where(atr > 0, atr, 1e-10)
        minus_di = 100 * minus_dm_smooth / np.where(atr > 0, atr, 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        adx_strong = adx > 20
        adx_weak = adx < 15
    else:
        adx = np.full(len(high_1d), np.nan)
        adx_strong = np.zeros(len(high_1d), dtype=bool)
        adx_weak = np.ones(len(high_1d), dtype=bool)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    volume_decay_aligned = align_htf_to_ltf(prices, df_1d, volume_decay.astype(float))
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    # TRIX zero-line crossover signals
    trix_above_zero = trix > 0
    trix_below_zero = trix < 0
    trix_cross_up = (trix_above_zero) & (~np.roll(trix_above_zero, 1))
    trix_cross_down = (trix_below_zero) & (~np.roll(trix_below_zero, 1))
    # Handle first element
    trix_cross_up[0] = False
    trix_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(trix[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(volume_decay_aligned[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TRIX bullish crossover + volume spike + strong ADX
            if (trix_cross_up[i] and 
                volume_spike_aligned[i] == 1.0 and 
                adx_strong_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX bearish crossover + volume spike + strong ADX
            elif (trix_cross_down[i] and 
                  volume_spike_aligned[i] == 1.0 and 
                  adx_strong_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX bearish crossover OR volume decay OR weak ADX
            if (trix_cross_down[i] or 
                volume_decay_aligned[i] == 1.0 or 
                adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX bullish crossover OR volume decay OR weak ADX
            if (trix_cross_up[i] or 
                volume_decay_aligned[i] == 1.0 or 
                adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals