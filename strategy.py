#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Trend_v2
Hypothesis: Uses daily Camarilla pivot levels with weekly trend filter and volume confirmation.
Long when weekly uptrend AND price breaks above daily H3 with volume spike.
Short when weekly downtrend AND price breaks below daily L3 with volume spike.
Designed for low trade frequency by requiring weekly trend alignment and volume confirmation.
Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Trend_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_ws = pd.Series(close_1w)
    ema20_w = close_ws.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Camarilla pivot levels (based on previous day)
    # Calculate for each day, then shift by 1 to avoid look-ahead
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first day values to NaN (no previous day)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, np.nan, range_val)
    
    H3 = prev_close + range_val * 1.1 / 4
    L3 = prev_close - range_val * 1.1 / 4
    
    # Align weekly and daily indicators
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume spike detection (volume > 1.5x 20-period average)
    volume_series = pd.Series(volume_1d)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    volume_spike = volume > (1.5 * vol_ma_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(ema20_w_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_w_aligned[i]
        weekly_downtrend = close[i] < ema20_w_aligned[i]
        
        # Camarilla breakout with volume confirmation
        long_breakout = (close[i] > H3_aligned[i]) and volume_spike[i]
        short_breakout = (close[i] < L3_aligned[i]) and volume_spike[i]
        
        # Exit conditions: reverse of entry or trend change
        exit_long = not weekly_uptrend or (close[i] < H3_aligned[i])
        exit_short = not weekly_downtrend or (close[i] > L3_aligned[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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