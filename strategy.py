#!/usr/bin/env python3
"""
1d_1wPivot_Breakout_Trend_Volume_v2
Hypothesis: Trade breakouts at weekly pivot levels (R1/S1) on daily timeframe with 1w trend filter and volume confirmation.
Weekly pivots act as strong support/resistance levels. Breakouts in direction of weekly trend with volume confirmation
should capture significant moves. Weekly pivot calculation provides fewer, more significant levels than daily pivots,
reducing trade frequency. Works in bull/bear markets by aligning with weekly trend direction.
Target: 15-25 trades/year on 1d timeframe.
"""

name = "1d_1wPivot_Breakout_Trend_Volume_v2"
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
    
    # === Weekly OHLC for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points from previous week's OHLC
    ph_w = df_1w['high'].values
    pl_w = df_1w['low'].values
    pc_w = df_1w['close'].values
    
    # Weekly Pivot Point (PP)
    pp_w = (ph_w + pl_w + pc_w) / 3.0
    # Weekly R1 and S1 (primary breakout levels)
    r1_w = pp_w + (ph_w - pl_w)
    s1_w = pp_w - (ph_w - pl_w)
    
    # Align to daily timeframe
    r1_1d = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # === Weekly Trend Filter (EMA34) ===
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume Filter (1.5x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(ema34_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume
            if (close[i] > r1_1d[i] and 
                close[i] > ema34_1d[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S1 with downtrend and volume
            elif (close[i] < s1_1d[i] and 
                  close[i] < ema34_1d[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly pivot (mean reversion)
            if close[i] < pp_w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above weekly pivot (mean reversion)
            if close[i] > pp_w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals