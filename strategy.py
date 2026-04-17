#!/usr/bin/env python3
"""
4h_Keltner_Channel_Squeeze_v1
Hypothesis: Keltner Channel width contraction (low volatility) followed by expansion with directional breakout captures trend initiation in both bull and bear markets. Uses 1d ADX for trend strength filter and volume surge for confirmation. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Keltner Channel (20, 2.0) ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(
        np.maximum(
            np.maximum(high - low, np.abs(high - np.roll(close, 1))),
            np.abs(low - np.roll(close, 1))
        )
    ).rolling(window=20, min_periods=20).mean().values
    atr[0] = high[0] - low[0]  # first ATR
    
    upper = ema20 + 2.0 * atr
    lower = ema20 - 2.0 * atr
    width = upper - lower
    
    # === Keltner Squeeze: width < 20-period average width ===
    avg_width = pd.Series(width).rolling(window=20, min_periods=20).mean().values
    squeeze = width < avg_width
    
    # === Breakout direction ===
    breakout_up = close > upper
    breakout_down = close < lower
    
    # === Volume confirmation: volume > 1.5 x 20-period average volume ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > 1.5 * avg_volume
    
    # === 1d ADX for trend strength filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_1d * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Signal generation ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    warmup = 60
    
    for i in range(warmup, n):
        if (np.isnan(ema20[i]) or np.isnan(atr[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry: only when coming out of squeeze with volume surge
        if position == 0:
            if (squeeze[i-1] and  # was in squeeze previous bar
                breakout_up[i] and 
                volume_surge[i] and 
                adx_1d_aligned[i] > 20):  # strong trend
                signals[i] = 0.25
                position = 1
                continue
            elif (squeeze[i-1] and  # was in squeeze previous bar
                  breakout_down[i] and 
                  volume_surge[i] and 
                  adx_1d_aligned[i] > 20):  # strong trend
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit: when price returns to middle (EMA20) or volatility drops
        elif position == 1:
            if close[i] <= ema20[i] or width[i] < avg_width[i] * 0.5:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] >= ema20[i] or width[i] < avg_width[i] * 0.5:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Channel_Squeeze_v1"
timeframe = "4h"
leverage = 1.0