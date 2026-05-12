#!/usr/bin/env python3
# 6h_Keltner_Channel_Breakout_ADX20_Filter
# Hypothesis: Keltner Channel breakout on 6h with ADX(14)>20 trend filter. 
# Uses ATR-based channel (EMA20 ± 2*ATR) to capture breakouts with trend confirmation.
# Long when price breaks above upper Keltner with ADX>20, short when breaks below lower Keltner with ADX>20.
# Exit when price crosses EMA20 (middle line) to avoid whipsaws. 
# Designed for low trade frequency (12-37/year) by requiring both breakout and trend strength.
# Works in bull/bear markets by following trend direction via ADX filter.

name = "6h_Keltner_Channel_Breakout_ADX20_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Calculate EMA20 (middle line) and ATR(10) for Keltner Channel
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    tr_series = pd.Series(tr)
    atr10 = tr_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channels
    upper_keltner = ema20 + 2 * atr10
    lower_keltner = ema20 - 2 * atr10

    # Calculate ADX(14) for trend strength filter
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, and TR
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    atr_14 = tr_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = plus_dm_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = minus_dm_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_14 / atr_14
    minus_di = 100 * minus_dm_14 / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx_series = pd.Series(dx)
    adx = dx_series.ewm(span=14, adjust=False, min_periods=14).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema20[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above upper Keltner with ADX > 20
            if close[i] > upper_keltner[i] and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Keltner with ADX > 20
            elif close[i] < lower_keltner[i] and adx[i] > 20:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below EMA20 (middle line)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above EMA20 (middle line)
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals