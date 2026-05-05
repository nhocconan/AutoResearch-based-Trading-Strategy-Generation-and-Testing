#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX trend filter + volume confirmation
# Elder Ray: Bull Power = high - EMA13, Bear Power = EMA13 - low
# Long when: Bull Power > 0 AND ADX(12h) > 25 AND volume > 1.5x 20-period MA
# Short when: Bear Power > 0 AND ADX(12h) > 25 AND volume > 1.5x 20-period MA
# Exit when: opposing Elder Ray power > 0 OR ADX(12h) < 20 (trend weak)
# Uses Elder Ray for momentum, ADX for regime, volume for conviction
# Timeframe: 6h, HTF: 12h for ADX. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_12hADX_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate EMA13 for Elder Ray on 6h
    if len(close) >= 13:
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema13 = np.full(n, np.nan)
    
    # Elder Ray components
    bull_power = high - ema13  # > 0 indicates bullish momentum
    bear_power = ema13 - low   # > 0 indicates bearish momentum
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    if len(high_12h) >= 14:
        # True Range
        tr1 = np.abs(high_12h[1:] - low_12h[1:])
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full(len(df_12h), np.nan)
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema13[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND ADX > 25 AND volume filter
            if (bull_power[i] > 0 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND ADX > 25 AND volume filter
            elif (bear_power[i] > 0 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power > 0 (momentum shift) OR ADX < 20 (trend weak)
            if (bear_power[i] > 0 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (momentum shift) OR ADX < 20 (trend weak)
            if (bull_power[i] > 0 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals