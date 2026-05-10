#!/usr/bin/env python3
# 6h_ADX_Alligator_Trend_Filter
# Hypothesis: Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) combined with ADX(14) filters whipsaw in sideways markets.
# Long when Lips > Teeth > Jaw (bullish alignment) + ADX > 25; Short when Lips < Teeth < Jaw (bearish alignment) + ADX > 25.
# Uses 1d trend (EMA50) for multi-timeframe alignment to avoid counter-trend trades.
# Designed for low-moderate trade frequency (target: 15-30 trades/year) with strong trend confirmation.

name = "6h_ADX_Alligator_Trend_Filter"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Williams Alligator (SMAs) ===
    # Jaw: 13-period SMMA (smoothed SMA)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # === ADX Calculation (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === Daily Trend Filter (EMA50) ===
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator (13), ADX (14*2=28), daily EMA (50)
    start_idx = max(13, 28, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment
        bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: bullish Alligator + strong ADX + daily uptrend
            if bullish_align and strong_trend and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator + strong ADX + daily downtrend
            elif bearish_align and strong_trend and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or ADX weakens
            if not bullish_align or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or ADX weakens
            if not bearish_align or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals