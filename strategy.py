#!/usr/bin/env python3
name = "12h_Donchian_20_Breakout_1dTrend_VolumeSpike"
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
    
    # Daily trend: EMA34 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel on 12h (upper/lower 20-period)
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: volume > 1.5 * 20-period SMA of volume
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_sma
    
    signals = np.zeros(n)
    position = 0  # 1 = long, -1 = short, 0 = flat
    
    start_idx = max(lookback, 34, 20)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + daily uptrend + volume spike
            if high[i] > donch_high[i] and close[i] > ema_34_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + daily downtrend + volume spike
            elif low[i] < donch_low[i] and close[i] < ema_34_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below lower Donchian or trend reversal
            if close[i] < donch_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above upper Donchian or trend reversal
            if close[i] > donch_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals