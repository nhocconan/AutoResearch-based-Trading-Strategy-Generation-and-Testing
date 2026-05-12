#!/usr/bin/env python3
"""
4h_1d_Alligator_MF_Signal_v1
Hypothesis: Bill Williams Alligator (13,8,5 SMAs) on 1d timeframe provides strong trend direction.
Enter long when price crosses above Alligator Jaw (13-bar SMA) with price > Teeth (8-bar SMA) > Lips (5-bar SMA),
enter short when price crosses below Jaw with price < Teeth < Lips.
Use 4h timeframe for execution with volume confirmation (>1.5x 20-bar average) and ATR volatility filter.
Only trade in trending markets (ADX(14) > 25 on 1d).
Exit when price crosses back below/above Jaw or ADX drops below 20.
Position size: 0.25 to limit drawdown in volatile markets.
Designed to work in both bull (trend following) and bear (strong downtrends) markets.
"""

name = "4h_1d_Alligator_MF_Signal_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # ATR for volatility filter (optional, can be used for position sizing but we keep fixed size)
    # TR calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d data for Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for slowest SMA (13) and ADX
        return np.zeros(n)
    
    # Alligator components on 1d
    close_1d = df_1d['close'].values
    # Jaw (13-bar SMA), Teeth (8-bar SMA), Lips (5-bar SMA)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # ADX calculation on 1d
    # +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # TR for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1_1d[0], tr2_1d[0], tr3_1d[0]])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    # Smoothed values
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    # Smooth +DM and -DM
    plus_dm_smoothed = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    plus_di = 100 * plus_dm_smoothed / np.where(atr_1d == 0, 1, atr_1d)
    minus_di = 100 * minus_dm_smoothed / np.where(atr_1d == 0, 1, atr_1d)
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trending market filter: ADX > 25
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # LONG: Price crosses above Jaw with proper Alligator alignment (Jaw < Teeth < Lips in uptrend)
            # Actually, in uptrend: Lips > Teeth > Jaw, price above all
            if (is_trending and 
                close[i] > jaw_aligned[i] and 
                close[i] > teeth_aligned[i] and 
                close[i] > lips_aligned[i] and
                lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below Jaw with proper Alligator alignment (Jaw > Teeth > Lips in downtrend)
            elif (is_trending and 
                  close[i] < jaw_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  close[i] < lips_aligned[i] and
                  lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Jaw OR ADX drops below 20 (trend weakening)
            if close[i] < jaw_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Jaw OR ADX drops below 20
            if close[i] > jaw_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals