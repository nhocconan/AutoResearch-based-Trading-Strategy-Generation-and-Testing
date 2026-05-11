#!/usr/bin/env python3
name = "4h_WeeklyTrend_12hVolumeSpike_1dPullback"
timeframe = "4h"
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
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 12h volume spike: volume > 1.5 * 20-period SMA of volume (on 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    vol_sma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_12h)
    vol_spike = volume_12h > 1.5 * vol_sma_12h_aligned
    
    # 1d pullback: close < 20-period EMA on daily
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_12h_aligned[i]) or np.isnan(ema_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + 12h volume spike + 1d pullback
            if close[i] > ema_34_1w_aligned[i] and vol_spike[i] and close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + 12h volume spike + 1d pullback
            elif close[i] < ema_34_1w_aligned[i] and vol_spike[i] and close[i] > ema_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly trend reversal or close above 1d EMA20
            if close[i] < ema_34_1w_aligned[i] or close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend reversal or close below 1d EMA20
            if close[i] > ema_34_1w_aligned[i] or close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals