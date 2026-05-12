#!/usr/bin/env python3
"""
4H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE_V2
Hypothesis: Refine the original by tightening entry conditions to reduce trade frequency and avoid overtrading.
Use the same core logic (R3/S3 breakout + volume spike + 1d EMA34 trend filter) but add:
- Minimum 4-bar hold time to prevent whipsaw exits
- Volume spike threshold increased to 2.0x (from 1.5x) for stronger confirmation
- Entry only when price closes beyond the level (not just intraday touch)
Target: 15-25 trades/year to stay well under the 400 total 4h trade limit.
Works in bull markets (breakouts continue) and bear markets (sharp reversals from S3/R3).
"""
name = "4H_CAMARILLA_R3_S3_BREAKOUT_1DVOLUMESPIKE_V2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 2.0 * 20-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R3 = prev_close + rang * 1.1 / 2
    S3 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(34, n):  # Start after warmup for EMA34
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # LONG: Break above R3 + volume spike + above 1d EMA34 (uptrend)
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Break below S3 + volume spike + below 1d EMA34 (downtrend)
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Only after minimum 4 bars AND (price re-enters OR trend reversal)
            if bars_since_entry >= 4 and ((close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Only after minimum 4 bars AND (price re-enters OR trend reversal)
            if bars_since_entry >= 4 and ((close[i] < R3_aligned[i] and close[i] > S3_aligned[i]) or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals