#!/usr/bin/env python3
# 12h_1W_1D_VolumeSpike_Camarilla_R1_S1_Breakout
# Hypothesis: Breakouts from weekly-derived Camarilla R1/S1 levels, confirmed by 12h volume spikes and aligned with 1d trend (price > EMA34 on daily).
# Works in bull markets via breakout continuation and in bear markets via mean-reversion from extremes (short at S1, long at R1).
# Uses 12h timeframe for execution to limit trade frequency (12-37 trades/year target).

name = "12h_1W_1D_VolumeSpike_Camarilla_R1_S1_Breakout"
timeframe = "12h"
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
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Camarilla R1/S1 levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly data for Camarilla R1/S1 levels (using previous week's OHLC)
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    rang_1w = prev_high_1w - prev_low_1w
    R1_1w = prev_close_1w + 1.1 * rang_1w * 1.1 / 4
    S1_1w = prev_close_1w - 1.1 * rang_1w * 1.1 / 4
    
    # Align weekly Camarilla levels to 12h timeframe
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA34
        if (np.isnan(R1_1w_aligned[i]) or 
            np.isnan(S1_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1_1w + volume spike + price above 1d EMA34 (uptrend)
            if (close[i] > R1_1w_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1_1w + volume spike + price below 1d EMA34 (downtrend)
            elif (close[i] < S1_1w_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous 1w H-L range (between S1 and R1) OR closes below 1d EMA34
            if (close[i] < R1_1w_aligned[i] and close[i] > S1_1w_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous 1w H-L range (between S1 and R1) OR closes above 1d EMA34
            if (close[i] < R1_1w_aligned[i] and close[i] > S1_1w_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals