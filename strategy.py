#!/usr/bin/env python3
"""
6h ADX + Williams Alligator with 1d Trend Filter
Long when Alligator bullish (Jaw < Teeth < Lips) and ADX > 25
Short when Alligator bearish (Jaw > Teeth > Lips) and ADX > 25
Exit when Alligator reverses or ADX < 20 (trend weakening)
Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
Designed to catch strong trends in both bull and bear markets with proper filtering
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Williams Alligator ===
    # Jaw: Blue line - 13-period SMMA smoothed 8 periods
    jaw_raw = pd.Series(high).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: Red line - 8-period SMMA smoothed 5 periods
    teeth_raw = pd.Series(low).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: Green line - 5-period SMMA smoothed 3 periods
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # === ADX (14) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 1d Trend Filter (EMA50 > EMA200 for bullish, < for bearish) ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(adx[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator turns bearish OR ADX < 20 (trend weakening)
            if (jaw[i] > teeth[i] or teeth[i] > lips[i]) or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR ADX < 20 (trend weakening)
            if (jaw[i] < teeth[i] or teeth[i] < lips[i]) or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish Alligator: Jaw < Teeth < Lips
            bullish_alligator = jaw[i] < teeth[i] and teeth[i] < lips[i]
            # Bearish Alligator: Jaw > Teeth > Lips
            bearish_alligator = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            if bullish_alligator and adx[i] > 25 and ema_50_aligned[i] > ema_200_aligned[i]:
                # Strong uptrend in bullish regime
                position = 1
                signals[i] = 0.25
            elif bearish_alligator and adx[i] > 25 and ema_50_aligned[i] < ema_200_aligned[i]:
                # Strong downtrend in bearish regime
                position = -1
                signals[i] = -0.25
    
    return signals