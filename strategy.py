#!/usr/bin/env python3
"""
6h_12h_Keltner_Channel_Breakout_Trend_Filter
Hypothesis: Keltner Channel (ATR-based) breakouts on 6h with 12h EMA50 trend filter and volume confirmation capture sustained moves in both bull and bear markets. The ATR-based bands adapt to volatility, reducing false breakouts in ranging markets while capturing true trends. Volume confirmation ensures breakouts have participation. Target: 15-40 trades/year per symbol.
"""

name = "6h_12h_Keltner_Channel_Breakout_Trend_Filter"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 6h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Keltner Channel on 6h: 20 EMA ± 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr = np.zeros(n)
    for i in range(10, n):
        atr[i] = np.mean(tr[i-10:i])
    
    # Keltner Bands
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values for current bar
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price breaks above Keltner upper, 12h uptrend, volume confirmation
            if close[i] > keltner_upper[i] and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Keltner lower, 12h downtrend, volume confirmation
            elif close[i] < keltner_lower[i] and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below 20 EMA or 12h trend turns down
            if close[i] < ema_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above 20 EMA or 12h trend turns up
            if close[i] > ema_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals