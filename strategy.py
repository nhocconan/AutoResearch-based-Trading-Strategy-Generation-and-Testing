#163122
#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as strong support/resistance. Breakouts above R1 or below S1 with volume confirmation and aligned with 1w EMA50 trend capture institutional moves. Weekly trend filter avoids counter-trend trades in chop. Designed for low-frequency, high-conviction trades on 12h timeframe.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    close_val = close
    r1 = close_val + (range_val * 1.1 / 12)
    s1 = close_val - (range_val * 1.1 / 12)
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R1, S1) for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + above 1w EMA50 (uptrend) + volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + below 1w EMA50 (downtrend) + volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 (failed breakout) or below 1w EMA50 (trend change)
            if (close[i] < s1_1d_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 (failed breakdown) or above 1w EMA50 (trend change)
            if (close[i] > r1_1d_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals