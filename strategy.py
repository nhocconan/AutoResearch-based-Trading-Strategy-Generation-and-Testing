#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 12h Supertrend combination for trend strength and direction
# ADX > 25 indicates strong trend (works in both bull/bear markets)
# 12h Supertrend provides trend direction (long when price > Supertrend, short when price < Supertrend)
# Volume confirmation: current volume > 1.5x 20-period MA to avoid false breakouts
# Entry: Long when ADX>25 AND close > 12h Supertrend AND volume spike
# Entry: Short when ADX>25 AND close < 12h Supertrend AND volume spike
# Exit: When ADX < 20 (trend weakening) OR price crosses Supertrend in opposite direction
# Uses ADX for trend strength (avoids choppy markets), Supertrend for direction, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ADX_12hSupertrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for ATR and Supertrend
        return np.zeros(n)
    
    # Calculate 12h ATR(10) for Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 12h Supertrend
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    supertrend = np.full(len(close_12h), np.nan)
    direction = np.full(len(close_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_10[i]) or np.isnan(close_12h[i]):
            continue
            
        if i == 1:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_12h[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    # Align 12h Supertrend and direction to 6h
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 6h ADX(14)
    if len(close) >= 14:
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr_14 == 0, np.nan, atr_14)
        di_minus = 100 * dm_minus_smooth / np.where(atr_14 == 0, np.nan, atr_14)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full(n, np.nan)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX>25 AND above 12h Supertrend (uptrend) AND volume spike
            if (adx[i] > 25 and 
                direction_aligned[i] == 1 and 
                close[i] > supertrend_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: ADX>25 AND below 12h Supertrend (downtrend) AND volume spike
            elif (adx[i] > 25 and 
                  direction_aligned[i] == -1 and 
                  close[i] < supertrend_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX < 20 (trend weakening) OR price crosses below Supertrend
            if adx[i] < 20 or close[i] <= supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX < 20 (trend weakening) OR price crosses above Supertrend
            if adx[i] < 20 or close[i] >= supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals