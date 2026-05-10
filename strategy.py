#!/usr/bin/env python3
# 1d_Range_Breakout_HTF_Trend
# Hypothesis: Trade daily breakouts from 20-period range when aligned with weekly trend and volume confirmation.
# Works in bull/bear by following weekly trend direction. Range breakouts capture momentum after consolidation.
# Volume filter ensures breakout conviction. Designed for low trade frequency (~15-25/year) to minimize fee drag.

name = "1d_Range_Breakout_HTF_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily range breakout: Donchian(20) channels
    period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period:i])
        lower[i] = np.min(low[i-period:i])
    
    # Weekly trend filter: EMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band, weekly uptrend, volume confirmation
            if close[i] > upper[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, weekly downtrend, volume confirmation
            elif close[i] < lower[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below lower band (range breakdown) or weekly trend turns down
            if close[i] < lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above upper band (range breakout) or weekly trend turns up
            if close[i] > upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals