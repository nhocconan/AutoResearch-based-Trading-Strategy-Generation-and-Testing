#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND close > 1d EMA34 AND volume > 1.5 * avg volume.
# Short when price breaks below Donchian lower channel AND close < 1d EMA34 AND volume > 1.5 * avg volume.
# Exit when price touches Donchian middle line (mean of upper/lower) or ATR-based stoploss (2 * ATR).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring confluence of breakout, trend, and volume.
# Donchian channels provide clear structure, EMA34 filters trend direction, volume confirms conviction.
# Effective in both bull and bear markets by capturing strong breakouts with trend and volume filters.

name = "4h_Donchian20_Breakout_1dEMA34_Volume_v1"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian(20) channels
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after Donchian period
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: break above upper channel, price > 1d EMA34, volume > 1.5 * avg
            if close[i] > upper_channel[i] and close[i] > ema34_1d_aligned[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
            # SHORT: break below lower channel, price < 1d EMA34, volume > 1.5 * avg
            elif close[i] < lower_channel[i] and close[i] < ema34_1d_aligned[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches middle channel OR ATR stoploss (2 * ATR below entry)
            if close[i] <= middle_channel[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: price touches middle channel OR ATR stoploss (2 * ATR above entry)
            if close[i] >= middle_channel[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals