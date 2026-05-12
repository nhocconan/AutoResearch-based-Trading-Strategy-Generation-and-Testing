#!/usr/bin/env python3
"""
4h_Equilibrium_Signal_Combined
Hypothesis: Combines equilibrium-based mean reversion with trend alignment to capture 
short-term reversals within the dominant trend. Uses 1h RSI for overbought/oversold 
signals, filtered by 4h Supertrend direction and 1d volume confirmation. 
Designed to work in both bull and bear markets by only taking mean-reversion trades 
in the direction of the higher timeframe trend. Targets 20-50 trades/year to avoid 
fee drag.
"""

name = "4h_Equilibrium_Signal_Combined"
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

    # Get 1h data for RSI (entry signal)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)

    close_1h = df_1h['close'].values
    # Calculate 1h RSI(14)
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)

    # Get 4h data for Supertrend (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate ATR(10) for Supertrend
    tr1 = np.abs(np.subtract(high_4h, low_4h))
    tr2 = np.abs(np.subtract(high_4h, np.roll(close_4h, 1)))
    tr3 = np.abs(np.subtract(low_4h, np.roll(close_4h, 1)))
    tr1[0] = tr2[0]  # First value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values

    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3 * atr_10)
    lower_band = hl2 - (3 * atr_10)
    
    # Initialize trend
    supertrend = np.zeros_like(close_4h)
    trend = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
    supertrend = np.where(trend == 1, lower_band, upper_band)
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values
        rsi = rsi_1h_aligned[i]
        st = supertrend_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol_1d = volume_1d[i // 24] if i >= 24 else volume_1d[0]  # Approximate 1d volume index

        # Skip if any data is NaN
        if np.isnan(rsi) or np.isnan(st) or np.isnan(vol_ma):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume confirmation: current 1d volume > 1.5x 20-day average
        vol_confirm = volume_1d[i // 24] > vol_ma_20_aligned[i] * 1.5 if i >= 24 else False

        if position == 0:
            # LONG: RSI < 30 (oversold) + price above Supertrend (uptrend) + volume confirmation
            if rsi < 30 and close[i] > st and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + price below Supertrend (downtrend) + volume confirmation
            elif rsi > 70 and close[i] < st and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (neutral) or price below Supertrend
            if rsi > 50 or close[i] < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (neutral) or price above Supertrend
            if rsi < 50 or close[i] > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals