#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 12h Camarilla pivot levels (from previous 12h bar)
    high_12h = high.copy()
    low_12h = low.copy()
    close_12h = close.copy()
    
    # Calculate pivot from previous 12h bar
    H = high_12h
    L = low_12h
    C = close_12h
    
    # Camarilla levels for previous bar
    R4 = C + ((H - L) * 1.1 / 2)
    R3 = C + ((H - L) * 1.1 / 4)
    R2 = C + ((H - L) * 1.1 / 6)
    R1 = C + ((H - L) * 1.1 / 12)
    S1 = C - ((H - L) * 1.1 / 12)
    S2 = C - ((H - L) * 1.1 / 6)
    S3 = C - ((H - L) * 1.1 / 4)
    S4 = C - ((H - L) * 1.1 / 2)
    
    # Shift to get previous bar's levels
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R1_prev[i]) or np.isnan(S1_prev[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume + 1d uptrend
            if close[i] > R1_prev[i] and volume_filter[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume + 1d downtrend
            elif close[i] < S1_prev[i] and volume_filter[i] and not trend_up[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below S1 or 1d trend down
            if close[i] < S1_prev[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above R1 or 1d trend up
            if close[i] > R1_prev[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals