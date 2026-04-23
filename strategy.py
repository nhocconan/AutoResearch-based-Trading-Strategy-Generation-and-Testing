#!/usr/bin/env python3
"""
Hypothesis: 6h ADX(14) + Volume Spike + EMA50 Trend Filter
Long when ADX > 25 (trending) AND volume > 2.0x 20-period average AND close > EMA50
Short when ADX > 25 (trending) AND volume > 2.0x 20-period average AND close < EMA50
Exit when ADX < 20 (trend weakens) OR volume < 1.5x average
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-35 trades/year per symbol.
ADX filters choppy markets, volume confirms institutional participation, EMA50 provides trend direction.
Works in both bull and bear markets by only taking strong trend trades with volume confirmation.
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
    
    # Load 1d data for ADX calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Current 6h close for price comparison
        price_6h = close[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Entry conditions: Strong trend + volume confirmation
            if (adx_aligned[i] > 25 and 
                volume[i] > 2.0 * vol_ma_val):
                if price_6h > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price_6h
                elif price_6h < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price_6h
        else:
            # Exit conditions: Trend weakening or volume drying up
            exit_signal = False
            
            if adx_aligned[i] < 20 or volume[i] < 1.5 * vol_ma_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ADX_Volume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0