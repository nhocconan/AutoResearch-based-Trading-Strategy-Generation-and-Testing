#!/usr/bin/env python3
"""
4h_1d_1w_PriceChannel_Breakout_TrendVol
Hypothesis: 4-hour breakouts from 1-day high/low channels with 1-week trend filter and volume confirmation.
Works in bull/bear markets by requiring 1-week trend alignment and volume spikes to avoid false breakouts.
Targets 25-40 trades per year (100-160 total over 4 years) to minimize fee drag.
"""

name = "4h_1d_1w_PriceChannel_Breakout_TrendVol"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for price channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Price channel from previous 1d high/low
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: only use previous day's data
    channel_high = prev_high
    channel_low = prev_low
    
    # Align channel to 4h timeframe (wait for 1d bar to close)
    channel_high_aligned = align_htf_to_ltf(prices, df_1d, channel_high)
    channel_low_aligned = align_htf_to_ltf(prices, df_1d, channel_low)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(channel_high_aligned[i]) or
            np.isnan(channel_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 1d high + volume spike + price above 1w EMA34
            if (close[i] > channel_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 1d low + volume spike + price below 1w EMA34
            elif (close[i] < channel_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters channel OR closes below 1w EMA34
            if (channel_low_aligned[i] <= close[i] <= channel_high_aligned[i]) or \
               close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters channel OR closes above 1w EMA34
            if (channel_low_aligned[i] <= close[i] <= channel_high_aligned[i]) or \
               close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals