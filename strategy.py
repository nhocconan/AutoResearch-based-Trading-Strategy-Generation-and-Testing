#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Use 4-hour Camarilla pivot levels (R1/S1) for breakout entries on 1h timeframe.
# Enter long on break above R1 with volume confirmation and 4h uptrend.
# Enter short on break below S1 with volume confirmation and 4h downtrend.
# Exit when price returns to the 4h pivot (PP) or trend reverses.
# This strategy uses higher timeframe (4h) for structure and lower timeframe (1h) for timing.
# By requiring confluence of level break, volume, and trend, we target 15-37 trades/year.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Session filter (08-20 UTC) reduces noise trades.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4-hour data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    # Calculate 4-hour Camarilla levels based on previous 4h bar's high, low, close
    ph = df_4h['high'].shift(1).values  # Previous 4h high
    pl = df_4h['low'].shift(1).values   # Previous 4h low
    pc = df_4h['close'].shift(1).values # Previous 4h close
    
    # Calculate pivot and ranges
    pivot = (ph + pl + pc) / 3
    range_val = ph - pl
    
    # Camarilla levels (R1, S1, and pivot for exit)
    r1 = pc + range_val * 1.1 / 12
    s1 = pc - range_val * 1.1 / 12
    pp = pivot  # Pivot point for exit
    
    # Align 4-hour levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp)
    
    # 4h trend filter: EMA21
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume confirmation: current volume > 1.5x average of last 4 periods (1 hour)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_21_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check trend alignment from 4h EMA21
        price_above_ema = close[i] > ema_21_aligned[i]
        price_below_ema = close[i] < ema_21_aligned[i]

        if position == 0:
            # LONG: break above R1 with volume, uptrend, and in session
            if close[i] > r1_aligned[i] and volume_ok[i] and price_above_ema and session_ok[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: break below S1 with volume, downtrend, and in session
            elif close[i] < s1_aligned[i] and volume_ok[i] and price_below_ema and session_ok[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to pivot or trend turns down
            if close[i] < pp_aligned[i] or close[i] < ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: return to pivot or trend turns up
            if close[i] > pp_aligned[i] or close[i] > ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals