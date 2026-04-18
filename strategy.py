#!/usr/bin/env python3
"""
6h_ADX_Trend_Strength_with_1dEMA_Filter
Hypothesis: Strong trends (ADX > 25) combined with 1d EMA50 direction provide reliable entries with controlled drawdowns in both bull and bear markets.
Uses 60% of capital on signals to balance return and risk. Targets 20-40 trades/year to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX (14-period) on 6h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    atr[0] = tr[0]
    plus_dm_smooth[0] = plus_dm[0] if len(plus_dm) > 0 else 0
    minus_dm_smooth[0] = minus_dm[0] if len(minus_dm) > 0 else 0
    
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_val = plus_dm[i-1] if i-1 < len(plus_dm) else 0
        minus_dm_val = minus_dm[i-1] if i-1 < len(minus_dm) else 0
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm_val) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm_val) / 14
    
    # Calculate +DI and -DI
    plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros(n)
    adx[0] = dx[0] if len(dx) > 0 else 0
    for i in range(1, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Trend strength: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for EMA and ADX
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema_50_1d_aligned[i]
        is_strong_trend = strong_trend[i]
        
        if position == 0:
            # Long: strong uptrend + price above 1d EMA50
            if is_strong_trend and price > ema50:
                signals[i] = 0.30
                position = 1
            # Short: strong downtrend + price below 1d EMA50
            elif is_strong_trend and price < ema50:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            signals[i] = 0.30
            # Exit: trend weakens OR price crosses below EMA50
            if not is_strong_trend or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.30
            # Exit: trend weakens OR price crosses above EMA50
            if not is_strong_trend or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Trend_Strength_with_1dEMA_Filter"
timeframe = "6h"
leverage = 1.0