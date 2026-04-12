#!/usr/bin/env python3
"""
1d_1w_Camarilla_Reversion_v1
Hypothesis: Uses weekly trend filter (price above/below weekly EMA20) with daily Camarilla pivot reversals.
Long when weekly uptrend AND price touches Camarilla L3 level; short when weekly downtrend AND price touches Camarilla H3 level.
Designed for low trade frequency by requiring weekly trend alignment and precise pivot level touches.
Works in bull via buying pullbacks in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Reversion_v1"
timeframe = "1d"
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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_ws = pd.Series(close_1w)
    ema20_w = close_ws.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily volume average for confirmation
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Calculate Camarilla pivot levels for previous day
    # Camarilla: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4, L4 = C - (H-L)*1.1/2
    # We use previous day's OHLC to calculate today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first day's values to avoid roll issues
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(ema20_w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_w_aligned[i]
        weekly_downtrend = close[i] < ema20_w_aligned[i]
        
        # Volume confirmation (above average)
        volume_confirm = volume[i] > volume_ma_aligned[i]
        
        # Camarilla level touches with small tolerance
        tol = 0.001  # 0.1% tolerance
        touch_h3 = abs(high[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < tol
        touch_l3 = abs(low[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < tol
        
        # Entry conditions
        long_setup = weekly_uptrend and touch_l3 and volume_confirm
        short_setup = weekly_downtrend and touch_h3 and volume_confirm
        
        # Exit when trend reverses or opposite touch
        exit_long = not weekly_uptrend or touch_h3
        exit_short = not weekly_downtrend or touch_l3
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
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