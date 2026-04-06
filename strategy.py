#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d EMA trend + volume confirmation + ATR stoploss.
Breakouts in direction of daily trend with volume filter capture strong moves.
ATR stoploss limits downside. Works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14297_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(20, 50)  # Donchian and EMA warmup
    
    for i in range(start, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or \
           np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: stoploss or opposite Donchian break
        if position == 1:  # long
            if close[i] <= entry_price - 2.5 * atr[i] or low[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short
            if close[i] >= entry_price + 2.5 * atr[i] or high[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Entry: Donchian breakout in direction of 1d EMA trend with volume
            long_breakout = high[i] > donchian_high[i]
            short_breakout = low[i] < donchian_low[i]
            uptrend = close[i] > ema_1d_aligned[i]
            downtrend = close[i] < ema_1d_aligned[i]
            
            if long_breakout and uptrend and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals