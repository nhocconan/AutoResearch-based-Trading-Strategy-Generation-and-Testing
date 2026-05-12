#/usr/bin/env python3
# 6h_12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 6-hour breakouts from 12-hour Camarilla R3/S3 levels with volume spike and 1-day EMA34 trend filter.
# R3/S3 represent strong reversal/breakout levels; breaks suggest continuation with institutional interest.
# In bull markets: breakouts above R3 capture momentum. In bear markets: breakdowns below S3 capture declines.
# Volume spike confirms institutional participation. 1-day EMA34 filters against counter-trend whipsaws.
# Targets 12-30 trades/year by requiring confluence of 12h level break, volume spike, and trend alignment.

name = "6h_12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h Camarilla R3 and S3 from previous 12h bar
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    rang_12h = prev_high_12h - prev_low_12h
    R3_12h = prev_close_12h + 1.1 * rang_12h * 3.0 / 4
    S3_12h = prev_close_12h - 1.1 * rang_12h * 3.0 / 4
    
    # Align 12h levels to 6h timeframe
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(R3_12h_aligned[i]) or 
            np.isnan(S3_12h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > R3_12h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < S3_12h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous 12h H-L range OR closes below daily EMA34
            if (close[i] < R3_12h_aligned[i] and close[i] > S3_12h_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous 12h H-L range OR closes above daily EMA34
            if (close[i] < R3_12h_aligned[i] and close[i] > S3_12h_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals