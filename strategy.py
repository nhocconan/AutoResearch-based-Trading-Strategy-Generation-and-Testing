#!/usr/bin/env python3
# 1h_4h_1D_Trend_Breakout_With_Volume_Spike
# Hypothesis: Use 1h for entry timing, 4h and 1d for trend and structure confirmation.
# Long when price breaks above 4h Donchian upper band with volume spike and 1d uptrend (price > EMA34).
# Short when price breaks below 4h Donchian lower band with volume spike and 1d downtrend (price < EMA34).
# Exit when price re-enters the 4h Donchian channel or 1d trend reverses.
# Designed for 1h timeframe with low trade frequency via multi-timeframe confirmation.
# Works in bull markets via breakout continuation and bear markets via mean-reversion from extremes.

name = "1h_4h_1D_Trend_Breakout_With_Volume_Spike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 4h data for Donchian breakout structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup for indicators
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4h Donchian high + volume spike + 1d uptrend
            if (close[i] > donch_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h Donchian low + volume spike + 1d downtrend
            elif (close[i] < donch_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters 4h Donchian channel OR 1d trend turns down
            if (close[i] < donch_high_aligned[i] and close[i] > donch_low_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters 4h Donchian channel OR 1d trend turns up
            if (close[i] < donch_high_aligned[i] and close[i] > donch_low_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals