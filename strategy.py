#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (R3/S3) breakout with volume confirmation
# and daily trend filter (EMA34). Enter long when price breaks above R3 with volume spike
# and daily EMA34 rising. Enter short when price breaks below S3 with volume spike
# and daily EMA34 falling. Uses volume spike (>1.5x 20-period average) to confirm
# breakout strength. Targets 50-150 total trades over 4 years (12-37/year) to minimize
# fee drag while capturing significant moves. Works in both bull and bear markets by
# following the daily trend direction for entries.

name = "12h_Camarilla_R3S3_Breakout_Volume_DailyTrend"
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
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Range = H - L
    rng = high - low
    # Camarilla levels
    R3 = close + (rng * 1.1 / 2)
    S3 = close - (rng * 1.1 / 2)
    # Shift to get previous day's levels
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    R3_prev[0] = 0
    S3_prev[0] = 0
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is invalid
        if (np.isclose(R3_prev[i], 0) or np.isclose(S3_prev[i], 0) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, volume spike, daily EMA34 rising
            if close[i] > R3_prev[i] and volume_spike[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, volume spike, daily EMA34 falling
            elif close[i] < S3_prev[i] and volume_spike[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or daily EMA34 turns down
            if close[i] < S3_prev[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or daily EMA34 turns up
            if close[i] > R3_prev[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals