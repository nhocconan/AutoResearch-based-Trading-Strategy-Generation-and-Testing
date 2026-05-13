#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation + ATR stoploss.
# Long when price breaks above Donchian upper band (20) AND close > 1d EMA50 AND volume > 1.5 * volume SMA(20).
# Short when price breaks below Donchian lower band (20) AND close < 1d EMA50 AND volume > 1.5 * volume SMA(20).
# Exit when price crosses Donchian middle band (10-period average of upper/lower) OR ATR-based stoploss (2 * ATR).
# Uses discrete position sizing (0.30) to limit fee churn and manage drawdown.
# Designed for moderate trade frequency (~20-50/year) by requiring confluence of breakout, trend, and volume.
# Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns with trend filter.

name = "4h_Donchian20_Breakout_1dEMA50_Volume_v1"
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
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5 * volume SMA(20)
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_sma
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = np.full(n, np.nan)
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(donchian_middle[i]) or np.isnan(volume_confirm[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Donchian upper AND close > 1d EMA50 AND volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
                entry_price[i] = close[i]
            # SHORT: price < Donchian lower AND close < 1d EMA50 AND volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Donchian middle OR price < entry_price - 2*ATR (stoploss)
            if close[i] < donchian_middle[i] or close[i] < entry_price[i-1] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.30
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: price > Donchian middle OR price > entry_price + 2*ATR (stoploss)
            if close[i] > donchian_middle[i] or close[i] > entry_price[i-1] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.30
                entry_price[i] = entry_price[i-1]
    
    return signals