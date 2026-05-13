#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike filter.
# Long when price breaks above Donchian upper channel AND 12h EMA50 is rising AND volume > 1.5 * avg volume.
# Short when price breaks below Donchian lower channel AND 12h EMA50 is falling AND volume > 1.5 * avg volume.
# Exit when price touches Donchian midpoint OR ATR-based stoploss (2 * ATR).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring confluence of breakout, trend, and volume.
# Donchian channels provide structure, EMA50 filters trend direction, volume spike confirms conviction.
# Effective in both bull and bear markets by capturing strong directional moves with filters.

name = "4h_Donchian20_Breakout_12hTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_12h_rising = np.zeros(n, dtype=bool)
    ema50_12h_falling = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(ema50_12h_aligned[i]) and not np.isnan(ema50_12h_aligned[i-1]):
            ema50_12h_rising[i] = ema50_12h_aligned[i] > ema50_12h_aligned[i-1]
            ema50_12h_falling[i] = ema50_12h_aligned[i] < ema50_12h_aligned[i-1]
    
    # Donchian(20) channels
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    # Average volume for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema50_12h_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper channel, 12h EMA50 rising, volume spike
            if close[i] > upper[i] and ema50_12h_rising[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
            # SHORT: price < lower channel, 12h EMA50 falling, volume spike
            elif close[i] < lower[i] and ema50_12h_falling[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < mid channel OR price < entry_price - 2*ATR (stoploss)
            if close[i] < mid[i] or close[i] < entry_price[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: price > mid channel OR price > entry_price + 2*ATR (stoploss)
            if close[i] > mid[i] or close[i] > entry_price[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals