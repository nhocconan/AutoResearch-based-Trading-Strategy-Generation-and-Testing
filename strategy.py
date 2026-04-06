#!/usr/bin/env python3
"""
1h Donchian Breakout + Volume Spike + 4h Trend Filter
Hypothesis: 1h momentum breakouts with volume confirmation in the direction of 4h trend capture short-term momentum while avoiding counter-trend whipsaws.
4h trend filter (EMA21) reduces false breakouts in choppy markets. Volume spike (2x average) confirms institutional participation.
Designed for 60-120 trades over 4 years (15-30/year) to minimize fee drag. Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14454_1h_donchian_vol_4htrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA21 for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Donchian channels (20-period)
    donchian_len = 20
    donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume spike filter (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_len, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + volume spike + 4h trend filter
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # 4h trend filter: only take longs in uptrend, shorts in downtrend
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1] if i > 0 else False
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1] if i > 0 else False
            
            if long_breakout and vol_spike[i] and trend_up:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_spike[i] and trend_down:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals