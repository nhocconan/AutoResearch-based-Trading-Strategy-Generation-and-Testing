#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Enters long when price breaks above 4h Donchian upper channel (20-bar high) with 12h bullish trend (close > EMA50) and volume > 1.8x MA20.
# Enters short when price breaks below 4h Donchian lower channel (20-bar low) with 12h bearish trend (close < EMA50) and volume > 1.8x MA20.
# Uses ATR-based stoploss: exits long when price drops below highest high since entry minus 2.5x ATR(14), exits short when price rises above lowest low since entry plus 2.5x ATR(14).
# Discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~20-50/year) to work in both bull and bear markets by requiring strong volume confirmation, trend alignment, and volatility-based stops.

name = "4h_Donchian_Breakout_12hTrend_Volume_ATRStop"
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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_high = 0.0  # highest high since entry for long
    entry_low = 0.0   # lowest low since entry for short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with 12h bullish trend and volume spike
            if close[i] > highest_high[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_high = high[i]  # initialize tracking variable
            # SHORT: Price breaks below Donchian lower channel with 12h bearish trend and volume spike
            elif close[i] < lowest_low[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_low = low[i]  # initialize tracking variable
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest high since entry for trailing stop
            entry_high = max(entry_high, high[i])
            # EXIT LONG: Price drops below highest high since entry minus 2.5x ATR
            if close[i] < (entry_high - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry for trailing stop
            entry_low = min(entry_low, low[i])
            # EXIT SHORT: Price rises above lowest low since entry plus 2.5x ATR
            if close[i] > (entry_low + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals