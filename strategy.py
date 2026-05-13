#!/usr/bin/env python3
"""
1d_1w_Volatility_Breakout_Trend_Follow
Hypothesis: In both bull and bear markets, volatility breakouts from ATR-based channels
with weekly trend alignment provide strong directional moves. Uses 1d ATR(20) for
channel width (1.5 * ATR) and 1w EMA(34) for trend filter. Volume confirmation
filters low-quality breakouts. Designed for low trade frequency (<25/year) to
minimize fee drag while capturing major trends.
"""

name = "1d_1w_Volatility_Breakout_Trend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(20) on 1d
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close_1d)
    for i in range(20, len(atr)):
        atr[i] = np.mean(tr[i-19:i+1])
    
    # Calculate upper/lower bands: close ± 1.5 * ATR
    upper_band = close_1d + 1.5 * atr
    lower_band = close_1d - 1.5 * atr
    
    # Align bands to daily timeframe (no shift needed as bands are based on closed 1d bar)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align 1w trend to daily
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.3 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: close breaks above upper band, weekly uptrend, volume confirmation
            if close[i] > upper and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: close breaks below lower band, weekly downtrend, volume confirmation
            elif close[i] < lower and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close breaks below lower band or weekly trend turns down
            if close[i] < lower or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close breaks above upper band or weekly trend turns up
            if close[i] > upper or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals