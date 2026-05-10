#!/usr/bin/env python3
"""
12h_Donchian_20_1dTrend_Volume
Hypothesis: Donchian(20) breakout on 12h timeframe with 1d EMA34 trend filter and volume confirmation.
In trending markets, price tends to break out of Donchian channels; in ranging markets, it reverts to mean.
Trend filter ensures we only trade in direction of higher timeframe trend.
Volume confirmation filters weak breakouts. Works in both bull (breakouts above upper channel) and bear (breakouts below lower channel).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_Donchian_20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Donchian Channel (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    if n >= donchian_period:
        for i in range(donchian_period-1, n):
            upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
            lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled to 12h)
        # Approximate 12h volume from 1d: 1d volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Price breaks above upper Donchian channel with uptrend and volume confirmation
            if close[i] > upper_channel[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian channel with downtrend and volume confirmation
            elif close[i] < lower_channel[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters Donchian Channel (below middle) or trend reversal
            mid_channel = (upper_channel[i] + lower_channel[i]) / 2
            if close[i] < mid_channel or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Donchian Channel (above middle) or trend reversal
            mid_channel = (upper_channel[i] + lower_channel[i]) / 2
            if close[i] > mid_channel or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals