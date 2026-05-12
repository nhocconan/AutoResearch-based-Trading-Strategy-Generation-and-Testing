#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout entries on 4h timeframe.
# Enter long on break above R1 with volume confirmation and daily trend filter (price > EMA34).
# Enter short on break below S1 with volume confirmation and daily trend filter (price < EMA34).
# Exit when price returns to daily pivot or trend reverses.
# This strategy targets breakouts in the direction of the daily trend with volume confirmation.
# Designed to work in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Targets 20-40 trades/year by requiring confluence of level break, volume, and trend.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate daily Camarilla levels
    # Based on previous day's high, low, close
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate pivot and ranges
    pivot = (ph + pl + pc) / 3
    range_val = ph - pl
    
    # Camarilla levels (R1/S1)
    r1 = pc + range_val * 1.1 / 6
    s1 = pc - range_val * 1.1 / 6
    pivot_level = pc  # Use close as pivot for exit
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x average of last 4 periods (1 day)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from daily EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]

        if position == 0:
            # LONG: break above R1 with volume and uptrend
            if close[i] > r1_aligned[i] and volume_ok[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume and downtrend
            elif close[i] < s1_aligned[i] and volume_ok[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to pivot or trend turns down
            if close[i] < pivot_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to pivot or trend turns up
            if close[i] > pivot_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals