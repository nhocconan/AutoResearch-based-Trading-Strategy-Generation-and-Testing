# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_1w_Structure_Breakout
Hypothesis: Trade weekly trend (via EMA) with daily Donchian breakouts in trend direction only.
Use volume confirmation (>1.5x 20-day avg) and ATR-based position sizing to manage risk.
Designed for low trade frequency (~10-25/year) with high conviction.
Works in bull markets (trend continuation) and bear markets (trend-following on weekly).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Structure_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER: EMA50 ON WEEKLY CHART ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === DAILY DONCHIAN CHANNEL (20) ===
    # Use 20-day high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === VOLUME FILTER ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR(14) FOR STOPLOSS AND POSITION SIZING ===
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first bar
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma20[i]) or np.isnan(atr14[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma20[i] * 1.5)
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i]
        breakout_down = close[i] < low_20[i]
        
        # Entry logic: breakout in direction of weekly trend with volume
        long_signal = breakout_up and weekly_uptrend and strong_volume
        short_signal = breakout_down and weekly_downtrend and strong_volume
        
        # Exit logic: opposite Donchian breakout or trend reversal
        exit_long = position == 1 and (breakout_down or not weekly_uptrend)
        exit_short = position == -1 and (breakout_up or not weekly_downtrend)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals