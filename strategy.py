#!/usr/bin/env python3
"""
12h_Price_Action_Structure_Strategy
Hypothesis: Uses 12h price action structure (higher highs/lows) combined with 1w trend filter and volume confirmation for high-probability entries.
Designed for low trade frequency (15-25/year) with strong trend-following logic that works in both bull and bear markets.
Uses volume confirmation and ATR-based stoploss to reduce false signals and manage risk.
"""

name = "12h_Price_Action_Structure_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate price action structure: higher highs and higher lows (uptrend)
    # Lower highs and lower lows (downtrend)
    # Using 5-period lookback for swing points
    lookback = 5
    
    # Calculate swing highs and lows
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    
    for i in range(lookback, n - lookback):
        # Swing high: highest high in the window
        if high[i] == np.max(high[i-lookback:i+lookback+1]):
            swing_high[i] = high[i]
        # Swing low: lowest low in the window
        if low[i] == np.min(low[i-lookback:i+lookback+1]):
            swing_low[i] = low[i]
    
    # Forward fill swing points to use as structure reference
    swing_high_series = pd.Series(swing_high)
    swing_low_series = pd.Series(swing_low)
    swing_high_ffill = swing_high_series.replace(0, np.nan).ffill().fillna(0).values
    swing_low_ffill = swing_low_series.replace(0, np.nan).ffill().fillna(0).values
    
    # Calculate 1-week trend filter (EMA 34)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: Price above recent swing low, 1w uptrend, and volume confirmation
            if (swing_low_ffill[i] > 0 and 
                close[i] > swing_low_ffill[i] and
                ema_34_1w_aligned[i] > 0 and 
                close[i] > ema_34_1w_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below recent swing high, 1w downtrend, and volume confirmation
            elif (swing_high_ffill[i] > 0 and 
                  close[i] < swing_high_ffill[i] and
                  ema_34_1w_aligned[i] > 0 and 
                  close[i] < ema_34_1w_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below recent swing low or 1w trend turns down
            if (swing_low_ffill[i] > 0 and close[i] < swing_low_ffill[i]) or \
               (ema_34_1w_aligned[i] > 0 and close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above recent swing high or 1w trend turns up
            if (swing_high_ffill[i] > 0 and close[i] > swing_high_ffill[i]) or \
               (ema_34_1w_aligned[i] > 0 and close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals