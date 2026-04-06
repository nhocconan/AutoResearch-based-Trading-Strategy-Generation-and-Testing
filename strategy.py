#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume + 1d EMA Trend Filter.
Hypothesis: Breakouts of the 20-period Donchian channel, confirmed by volume and 1d EMA trend,
provide directional edges. Works in bull (breakouts up) and bear (breakouts down) markets.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14320_4h_donchian20_volume_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA for daily trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.5x average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
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
    start = donchian_period + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price returns to lower channel OR stoploss
            if close[i] <= lower_channel[i] or close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to upper channel OR stoploss
            if close[i] >= upper_channel[i] or close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + 1d EMA trend filter
            long_breakout = close[i] > upper_channel[i-1]  # Break above previous high
            short_breakout = close[i] < lower_channel[i-1]  # Break below previous low
            trend_up = ema_1d_aligned[i] > ema_1d_aligned[i-1]  # Daily EMA rising
            trend_down = ema_1d_aligned[i] < ema_1d_aligned[i-1]  # Daily EMA falling
            
            if long_breakout and vol_confirm[i] and trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_confirm[i] and trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals