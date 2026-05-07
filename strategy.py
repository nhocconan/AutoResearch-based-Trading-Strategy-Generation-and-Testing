#!/usr/bin/env python3
name = "6h_ADX_Supertrend_DualTF_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ADX for regime filter
    # Calculate True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = dx.rolling(window=14, min_periods=14).mean().values
    
    # Daily Supertrend (ATR=10, multiplier=3)
    atr_10 = tr.rolling(window=10, min_periods=10).mean()
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    upper_band = hl2 + 3 * atr_10
    lower_band = hl2 - 3 * atr_10
    
    supertrend = np.full(len(df_1d), np.nan)
    direction = np.full(len(df_1d), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_1d)):
        if np.isnan(atr_10.iloc[i-1]) or np.isnan(upper_band.iloc[i-1]) or np.isnan(lower_band.iloc[i-1]):
            continue
            
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
    
        if direction[i] == 1:
            supertrend[i] = lower_band.iloc[i]
        else:
            supertrend[i] = upper_band.iloc[i]
    
    # Align daily indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # 6h ATR for volatility filter
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 14)  # Wait for ADX and Supertrend
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 and Supertrend alignment
        strong_trend = adx_aligned[i] > 25
        uptrend_aligned = close[i] > supertrend_aligned[i]
        downtrend_aligned = close[i] < supertrend_aligned[i]
        
        if position == 0:
            # Long: strong uptrend
            if strong_trend and uptrend_aligned:
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend
            elif strong_trend and downtrend_aligned:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: trend weakens or reverses
            if not strong_trend or not uptrend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: trend weakens or reverses
            if not strong_trend or not downtrend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s ADX + Supertrend dual timeframe trend following
# - Uses daily ADX (>25) to identify strong trending regimes (works in bull/bear)
# - Daily Supertrend provides trend direction entry signal
# - Only enters when both trend strength and direction align
# - Avoids whipsaws by requiring strong trend confirmation
# - Works in both bull (ADX>25 + uptrend) and bear (ADX>25 + downtrend)
# - Position size 0.25 limits drawdown during choppy periods
# - Targets 15-30 trades/year, avoiding excessive fee drag