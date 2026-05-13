#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w trend filter (EMA50) and volume spike (1.5x MA20).
# Enters long when price breaks above Donchian upper channel with 1w bullish trend (close > EMA50) and volume > 1.5x MA20.
# Enters short when price breaks below Donchian lower channel with 1w bearish trend (close < EMA50) and volume > 1.5x MA20.
# Uses ATR-based trailing stop: exits long when price drops 2.5*ATR from highest high since entry, exits short when price rises 2.5*ATR from lowest low since entry.
# Position sizing: 0.25 (25% of capital) to balance return and drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of breakout, trend, and volume.
# Works in both bull and bear markets: 1w trend filter ensures alignment with higher timeframe direction,
# while Donchian breakouts capture strong momentum moves and volume confirmation reduces false signals.

name = "12h_Donchian_Breakout_1wTrend_Volume_ATR"
timeframe = "12h"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with 1w bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # SHORT: Price breaks below Donchian lower channel with 1w bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest high since entry
            if close[i] > highest_high_since_entry:
                highest_high_since_entry = close[i]
            # EXIT LONG: Price drops 2.5*ATR from highest high since entry
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            if close[i] < lowest_low_since_entry:
                lowest_low_since_entry = close[i]
            # EXIT SHORT: Price rises 2.5*ATR from lowest low since entry
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals