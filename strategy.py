#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_ewo_oscillator_v1
# Uses Elder's Wave Oscillator (EWO) on 1d chart: EWO = (5-period SMA - 34-period SMA) / (14-period ATR) * 100
# Long when EWO > 3 (bullish momentum), short when EWO < -3 (bearish momentum)
# Requires volume > 1.5x 20-period average and ADX > 25 (trending market)
# Designed to capture strong trends in both bull and bear markets with low trade frequency
name = "4h_1d_ewo_oscillator_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EWO calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need at least 34 for SMA34
        return np.zeros(n)
    
    # Calculate EWO components on 1d
    sma5 = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
    sma34 = pd.Series(df_1d['close']).rolling(window=34, min_periods=34).mean().values
    # True Range for ATR
    tr1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # EWO = (SMA5 - SMA34) / ATR14 * 100
    ewo = (sma5 - sma34) / atr14 * 100
    # Handle division by zero or invalid ATR
    ewo = np.where(np.isnan(atr14) | (atr14 == 0), 0, ewo)
    
    # Align EWO to 4h timeframe
    ewo_aligned = align_htf_to_ltf(prices, df_1d, ewo)
    
    # Volume confirmation: volume > 1.5 * 20-period average (on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter: ADX > 25 indicates trending market
    # Calculate ADX components
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_adx = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(np.concatenate([[0], plus_dm])).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(np.concatenate([[0], minus_dm])).rolling(window=14, min_periods=14).mean().values
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_adx
    minus_di = 100 * minus_dm_smooth / atr_adx
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx = np.where(np.isnan(atr_adx) | (atr_adx == 0), 0, adx)
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    adx_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # start after warmup for SMA34
        # Skip if EWO not ready
        if np.isnan(ewo_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and ADX filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: EWO > 3 (bullish momentum)
        if ewo_aligned[i] > 3.0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: EWO < -3 (bearish momentum)
        elif ewo_aligned[i] < -3.0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: EWO returns to neutral zone (-3 to 3)
        elif -3.0 <= ewo_aligned[i] <= 3.0 and position != 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals