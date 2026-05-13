#!/usr/bin/env python3
"""
4h_ADX_Trend_With_Volume_Confirmation
Hypothesis: Trend following with ADX filter and volume confirmation works in both bull and bear markets by capturing strong trends while avoiding choppy periods. ADX > 25 filters for trending markets, volume > 1.5x average confirms institutional participation. Designed for low trade frequency (20-40/year) with clear entry/exit rules.
"""

name = "4h_ADX_Trend_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Calculate ADX (14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(low, 1)), abs(low - np.roll(high, 1))))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).values
    
    # Calculate EMA (20) for trend direction
    ema = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1-day trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: ADX > 25 (trending), price above EMA20, volume confirmation, and above 1-day EMA50
            if adx[i] > 25 and close[i] > ema[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: ADX > 25 (trending), price below EMA20, volume confirmation, and below 1-day EMA50
            elif adx[i] > 25 and close[i] < ema[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 (losing trend) or price crosses below EMA20
            if adx[i] < 20 or close[i] < ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 (losing trend) or price crosses above EMA20
            if adx[i] < 20 or close[i] > ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals