#!/usr/bin/env python3
name = "4h_WilliamsVixFix_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Vix Fix: measures volatility spike at market bottoms
    # WVF = ( (Highest Close in period - Low) / Highest Close in period ) * 100
    # Higher WVF = higher volatility = potential reversal point
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - low) / highest_close) * 100
    
    # Volume spike confirmation: current volume > 2.0 x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Trend filter: use 50-period EMA on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20) + 10
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(wvf[i]) or np.isnan(vol_ma20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: High WVF (volatility spike) + volume spike + price below EMA (oversold in downtrend)
            if wvf[i] > 80 and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit conditions: WVF normalizes OR price crosses above EMA (trend resumption)
            if wvf[i] < 40 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals