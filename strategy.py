#!/usr/bin/env python3
# 4h_Donchian20_Breakout_VolumeSpike_TrendFilter_ATRStop
# Hypothesis: 4-hour breakouts from 20-period Donchian channels with volume spike confirmation and 1-week EMA trend filter.
# In bull markets, weekly uptrend supports long breakouts above upper band; in bear markets, weekly downtrend supports short breakdowns below lower band.
# Volume spike ensures institutional participation, reducing false breakouts. ATR-based stop loss manages risk.
# Targets 20-40 trades per year by requiring confluence of weekly trend, Donchian breakout, and volume spike.

name = "4h_Donchian20_Breakout_VolumeSpike_TrendFilter_ATRStop"
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
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volatility filter: ATR(14) for stop loss later
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + price above weekly EMA34 (weekly uptrend)
            if (close[i] > highest_high[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + price below weekly EMA34 (weekly downtrend)
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel OR closes below weekly EMA34 OR ATR-based stop
            if (close[i] < highest_high[i] and close[i] > lowest_low[i]) or \
               close[i] < ema_34_1w_aligned[i] or \
               (atr[i] > 0 and close[i] < highest_high[i] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel OR closes above weekly EMA34 OR ATR-based stop
            if (close[i] < highest_high[i] and close[i] > lowest_low[i]) or \
               close[i] > ema_34_1w_aligned[i] or \
               (atr[i] > 0 and close[i] > lowest_low[i] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals