# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_v1
Strategy: 4-hour Camarilla pivot breakout with daily trend filter and volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Price breaking above/below Camarilla pivot levels (H3/L3) on 4h timeframe,
            confirmed by daily trend (EMA50 > EMA200 for long, EMA50 < EMA200 for short)
            and volume spike, captures institutional breakouts. Works in bull/bear
            by aligning with higher timeframe trend while using intraday precision.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_uptrend = ema50_1d > ema200_1d  # Uptrend when EMA50 > EMA200
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    
    # Calculate Camarilla pivot levels for each 4h bar
    # Using previous bar's OHLC (standard pivot calculation)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to current to avoid NaN
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    H4 = pivot + (range_hl * 1.1 / 2)
    L4 = pivot - (range_hl * 1.1 / 2)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d[i]) or np.isnan(ema200_1d[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(H3[i]) or np.isnan(L3[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_long = close[i] > H3[i] and vol_spike[i]
        breakout_short = close[i] < L3[i] and vol_spike[i]
        
        # Exit conditions: reverse breakout or volume drop
        exit_long = position == 1 and (close[i] < pivot[i] or not vol_spike[i])
        exit_short = position == -1 and (close[i] > pivot[i] or not vol_spike[i])
        
        # Trading logic: only trade in direction of daily trend
        if breakout_long and daily_uptrend_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and not daily_uptrend_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals