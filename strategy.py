#!/usr/bin/env python3
# 1h_4h_1D_Camarilla_R1_S1_Breakout_TrendVolume
# Hypothesis: 1-hour breakouts from daily-derived Camarilla R1/S1 levels with 4-hour trend filter and volume spike confirmation.
# Uses 4h trend and daily levels for signal direction, 1h for entry timing precision.
# Volume spike ensures institutional participation, reducing false breakouts.
# Targets 15-30 trades per year by requiring confluence of 4h trend, daily level break, and volume spike.
# Session filter (08-20 UTC) reduces noise trades outside active market hours.

name = "1h_4h_1D_Camarilla_R1_S1_Breakout_TrendVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Daily Camarilla R1 and S1 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.0 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.0 / 4
    
    # Align daily levels to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 4h EMA34 (4h uptrend)
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 4h EMA34 (4h downtrend)
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below 4h EMA34
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above 4h EMA34
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals