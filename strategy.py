#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX regime filter + volume confirmation
# Long when: 6h Bull Power > 0 AND 12h ADX > 25 (trending) AND 6h volume > 1.5x 20-period MA
# Short when: 6h Bear Power < 0 AND 12h ADX > 25 (trending) AND 6h volume > 1.5x 20-period MA
# Exit when: Elder Power crosses zero OR ADX < 20 (range) OR opposite signal occurs
# Uses Elder Ray for trend strength, ADX for regime confirmation, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

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
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
        bull_power = high - ema_13
        bear_power = low - ema_13
    else:
        ema_13 = np.full(n, np.nan)
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr1 = high_12h - low_12h
        tr2 = np.abs(high_12h - np.roll(close_12h, 1))
        tr3 = np.abs(low_12h - np.roll(close_12h, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        
        # Directional Movement
        up_move = high_12h - np.roll(high_12h, 1)
        down_move = np.roll(low_12h, 1) - low_12h
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    else:
        adx = np.full(len(df_12h), np.nan)
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND ADX > 25 (trending) AND volume filter
            if (bull_power[i] > 0 and 
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND ADX > 25 (trending) AND volume filter
            elif (bear_power[i] < 0 and 
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses zero OR ADX < 20 (range) OR Bear Power > 0
            if (bull_power[i] <= 0 or adx_aligned[i] < 20 or bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses zero OR ADX < 20 (range) OR Bull Power < 0
            if (bear_power[i] >= 0 or adx_aligned[i] < 20 or bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals