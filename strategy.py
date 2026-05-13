#!/usr/bin/env python3
"""
6h_ADX_Trend_Riding
Hypothesis: ADX > 25 identifies trending markets. Ride trends with +DI/-DI crossovers, using 1d trend filter to avoid counter-trend whipsaws. Exit when trend weakens (ADX < 20) or opposite crossover. Works in both bull and bear by capturing sustained moves.
Target: 15-30 trades/year per symbol.
"""

name = "6h_ADX_Trend_Riding"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX(14) calculation
    period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(period*2, n):
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        uptrend_1d_filt = uptrend_1d_aligned[i]
        downtrend_1d_filt = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: ADX > 25, +DI crosses above -DI, 1d uptrend filter
            if adx_val > 25 and plus_di_val > minus_di_val and plus_di[i-1] <= minus_di[i-1] and uptrend_1d_filt:
                signals[i] = 0.25
                position = 1
            # SHORT: ADX > 25, -DI crosses above +DI, 1d downtrend filter
            elif adx_val > 25 and minus_di_val > plus_di_val and minus_di[i-1] <= plus_di[i-1] and downtrend_1d_filt:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 (trend weak) or -DI crosses above +DI
            if adx_val < 20 or (minus_di_val > plus_di_val and minus_di[i-1] <= plus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 (trend weak) or +DI crosses above -DI
            if adx_val < 20 or (plus_di_val > minus_di_val and plus_di[i-1] <= minus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals