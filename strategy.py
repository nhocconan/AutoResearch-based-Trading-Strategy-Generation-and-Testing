#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_Trend_Volume
Hypothesis: Daily breakouts at weekly Camarilla R1/S1 levels with 1-week EMA34 trend filter and volume confirmation.
Trades only in direction of weekly trend to avoid counter-trend whipsaws. Uses weekly timeframe for trend,
daily for entry/exit, reducing trade frequency to 10-25/year. Works in bull/bear markets by aligning with
higher timeframe trend. Volume confirmation filters false breakouts. Low trade frequency minimizes fee drag.
"""

name = "1d_1w_Camarilla_R1S1_Breakout_Trend_Volume"
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
    
    # === Weekly Data for Trend Filter (EMA34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Weekly Data for Camarilla Levels (previous week's OHLC) ===
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ph_1w = df_1w['high'].values  # Previous week's high
    pl_1w = df_1w['low'].values   # Previous week's low
    pc_1w = df_1w['close'].values # Previous week's close
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1w = pc_1w + (ph_1w - pl_1w) * 1.1 / 12
    s1_1w = pc_1w - (ph_1w - pl_1w) * 1.1 / 12
    
    # Align to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Volume Filter (1.5x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA34 and weekly data)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above weekly R1 with uptrend and volume
            if (close[i] > r1_1w_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below weekly S1 with downtrend and volume
            elif (close[i] < s1_1w_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly S1 (mean reversion to opposite level)
            if close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above weekly R1 (mean reversion to opposite level)
            if close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals