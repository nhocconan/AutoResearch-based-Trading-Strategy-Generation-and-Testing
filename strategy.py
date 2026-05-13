#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_R4_S4_Breakout_Trend_Volume
Hypothesis: On the 6h timeframe, breakouts beyond the weekly Camarilla R4/S4 levels
with daily trend alignment and volume confirmation capture strong momentum moves.
In bull markets, R4 breakouts with daily uptrend signal longs; in bear markets,
S4 breakdowns with daily downtrend signal shorts. Weekly levels provide stronger
support/resistance than daily, reducing false breakouts. Volume filters ensure
breakouts have participation. Target: 15-35 trades/year per symbol.
"""

name = "6h_1w_1d_Camarilla_R4_S4_Breakout_Trend_Volume"
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
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation (R4, S4)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    # Using the standard Camarilla formula where R4/S4 are the strongest levels
    width_1w = high_1w - low_1w
    cam_r4 = close_1w + width_1w * 1.1 / 2
    cam_s4 = close_1w - width_1w * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe (wait for weekly bar to close)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1w, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1w, cam_s4)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align daily trend to 6h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.8 * 20-period average (higher threshold for fewer trades)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values for current bar
        r4 = cam_r4_aligned[i]
        s4 = cam_s4_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price breaks above weekly R4, daily uptrend, volume confirmation
            if close[i] > r4 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below weekly S4, daily downtrend, volume confirmation
            elif close[i] < s4 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below weekly S4 or daily trend turns down
            if close[i] < s4 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above weekly R4 or daily trend turns up
            if close[i] > r4 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals