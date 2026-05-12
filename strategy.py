#!/usr/bin/env python3
# 4h_12h_1d_Donchian_Breakout_VolumeTrend_v2
# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation. Targets breakouts in the direction of the 12h trend, with volume confirming institutional participation. Uses 1d ATR for volatility filtering to avoid choppy markets. Designed for fewer trades (<50/year) to minimize fee drag and improve generalization.

name = "4h_12h_1d_Donchian_Breakout_VolumeTrend_v2"
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
    
    # Volume spike: >2.0x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for ATR filter (avoid choppy markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # ATR(14) on daily
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 4h Donchian(20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid trading when ATR is too low (choppy market)
        if atr_14_1d_aligned[i] < 0.01 * close[i]:  # less than 1% of price
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike + price above 12h EMA34
            if (close[i] > highest_high[i] and 
                volume_spike[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike + price below 12h EMA34
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian high OR closes below 12h EMA34
            if (close[i] < highest_high[i]) or \
               close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian low OR closes above 12h EMA34
            if (close[i] > lowest_low[i]) or \
               close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals