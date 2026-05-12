#!/usr/bin/env python3

# 4h_12h_1d_MultiLevel_Breakout_Trend_Filter
# Hypothesis: Combines 1d Camarilla R4/S4 with 12h Donchian breakouts and volume confirmation.
# Long: Price breaks above BOTH 1d R4 and 12h Donchian high + 12h uptrend + volume spike.
# Short: Price breaks below BOTH 1d S4 and 12h Donchian low + 12h downtrend + volume spike.
# Dual confirmation reduces false breakouts while maintaining trend alignment.
# Designed for low trade frequency (target 30-80 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by requiring alignment between daily support/resistance and 12h trend/momentum.

name = "4h_12h_1d_MultiLevel_Breakout_Trend_Filter"
timeframe = "4h"
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
    
    # Volume spike: >1.8x 30-period average (stricter filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R4/S4 levels
    camarilla_r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # 12h data for Donchian channels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if (np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above BOTH 1d R4 AND 12h Donchian high + 12h uptrend + volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below BOTH 1d S4 AND 12h Donchian low + 12h downtrend + volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 12h Donchian low OR closes below 12h EMA50
            if (close[i] < donchian_low_aligned[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above 12h Donchian high OR closes above 12h EMA50
            if (close[i] > donchian_high_aligned[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals