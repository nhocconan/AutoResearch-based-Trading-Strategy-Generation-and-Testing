#!/usr/bin/env python3
"""
4h_1d_1w_WeeklyDonchianBreakout_TrendVol_v1
Hypothesis: Weekly Donchian breakout (based on weekly price action) with 1-day trend filter and volume spike confirmation.
This strategy targets 4h timeframe for balance between trade frequency and signal quality. It uses weekly Donchian
channels for breakout signals, 1-day EMA for trend filtering, and volume spikes to confirm breakout strength.
The strategy aims to capture major trend moves while avoiding false breakouts in sideways markets. It should work
in both bull and bear markets due to the trend filter and volume confirmation, which help avoid counter-trend trades.
"""

name = "4h_1d_1w_WeeklyDonchianBreakout_TrendVol_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1w data for weekly Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period) from previous weekly bar
    # Using highest high and lowest low over the past 20 weekly bars
    # Avoid look-ahead: only use previous weekly data
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().shift(1).values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align weekly Donchian levels to 4h timeframe (wait for weekly bar to close)
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(high_20w_aligned[i]) or
            np.isnan(low_20w_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high + volume spike + price above 1d EMA34
            if (close[i] > high_20w_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + volume spike + price below 1d EMA34
            elif (close[i] < low_20w_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters weekly Donchian range OR closes below 1d EMA34
            if (close[i] > low_20w_aligned[i] and close[i] < high_20w_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters weekly Donchian range OR closes above 1d EMA34
            if (close[i] > low_20w_aligned[i] and close[i] < high_20w_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals