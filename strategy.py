#!/usr/bin/env python3
# 12h_1W_1D_Camarilla_R3S3_Breakout_TrendVol
# Hypothesis: 12-hour breakouts from weekly and daily Camarilla R3/S3 levels with weekly trend filter and volume spike confirmation.
# Only takes long when price breaks above R3 with volume spike and weekly uptrend, short when breaks below S3 with volume spike and weekly downtrend.
# Uses tight entry conditions (trend + volume + level break) to target 12-37 trades per year, avoiding overtrading.
# Works in bull markets via trend-following breaks and in bear markets via counter-trend reversals at extreme weekly/daily levels.

name = "12h_1W_1D_Camarilla_R3S3_Breakout_TrendVol"
timeframe = "12h"
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
    
    # Weekly data for trend filter and higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for Camarilla R3 and S3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for additional trend confirmation
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla R3 and S3 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R3_1d = prev_close_1d + 1.1 * rang_1d * 3.0 / 4
    S3_1d = prev_close_1d - 1.1 * rang_1d * 3.0 / 4
    
    # Align daily levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above weekly EMA34 (weekly uptrend) + price above daily EMA34
            if (close[i] > R3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i] and
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below weekly EMA34 (weekly downtrend) + price below daily EMA34
            elif (close[i] < S3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i] and
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below weekly EMA34
            if (close[i] < R3_1d_aligned[i] and close[i] > S3_1d_aligned[i]) or \
               close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above weekly EMA34
            if (close[i] < R3_1d_aligned[i] and close[i] > S3_1d_aligned[i]) or \
               close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals