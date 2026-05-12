#!/usr/bin/env python3
# 12h_1w_1d_Momentum_Structure_Breakout
# Hypothesis: Combines 1-week trend (EMA34), 1-day momentum (RSI), and 12-hour breakout structure.
# Uses weekly EMA for trend direction, daily RSI for momentum filter, and breaks of 12h swing highs/lows
# for entry timing. Volume confirmation (>1.5x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<200 total 12h trades) to minimize fee drag.
# Works in bull/bear markets by following weekly trend while using daily momentum and 12h structure.

name = "12h_1w_1d_Momentum_Structure_Breakout"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly data for trend direction (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for momentum (RSI14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 12h swing points for entry/exit structure
    swing_high_12h = np.zeros(len(high), dtype=bool)
    swing_low_12h = np.zeros(len(low), dtype=bool)
    
    for i in range(1, len(high)-1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            swing_high_12h[i] = True
        if low[i] < low[i-1] and low[i] < low[i+1]:
            swing_low_12h[i] = True
    
    # Calculate 12h swing high and low levels
    last_swing_high_12h = np.full(len(high), np.nan)
    last_swing_low_12h = np.full(len(low), np.nan)
    
    last_high_12h = np.nan
    last_low_12h = np.nan
    
    for i in range(len(high)):
        if swing_high_12h[i]:
            last_high_12h = high[i]
        if swing_low_12h[i]:
            last_low_12h = low[i]
        last_swing_high_12h[i] = last_high_12h
        last_swing_low_12h[i] = last_low_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(last_swing_high_12h[i]) or
            np.isnan(last_swing_low_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weekly uptrend + Daily momentum + 12h breakout above swing high + volume spike
            if (close[i] > ema_34_1w_aligned[i] and  # Weekly uptrend
                rsi_14_1d_aligned[i] > 50 and       # Daily bullish momentum
                close[i] > last_swing_high_12h[i] and  # Break above 12h swing high
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + Daily momentum + 12h breakdown below swing low + volume spike
            elif (close[i] < ema_34_1w_aligned[i] and  # Weekly downtrend
                  rsi_14_1d_aligned[i] < 50 and       # Daily bearish momentum
                  close[i] < last_swing_low_12h[i] and  # Break below 12h swing low
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns bearish OR price breaks below 12h swing low
            if (close[i] < ema_34_1w_aligned[i]) or \
               (close[i] < last_swing_low_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns bullish OR price breaks above 12h swing high
            if (close[i] > ema_34_1w_aligned[i]) or \
               (close[i] > last_swing_high_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals