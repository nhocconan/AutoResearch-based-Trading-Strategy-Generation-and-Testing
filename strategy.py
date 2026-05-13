#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band breakout with 1d volume spike and 1w trend filter.
# Long when price breaks above upper BB(20,2) AND 1d volume > 2.0 * 20-period average volume AND 1w close > 1w EMA50.
# Short when price breaks below lower BB(20,2) AND 1d volume > 2.0 * 20-period average volume AND 1w close < 1w EMA50.
# Exit when price returns to middle BB (20-period SMA).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing volatility expansion in trending higher timeframes.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_BB_Breakout_1dVolumeSpike_1wTrendFilter_v1"
timeframe = "6h"
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
    
    # Calculate Bollinger Bands (20,2) on primary timeframe
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20-period SMA for exit
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB warmup
        # Skip if any required data is NaN
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper BB AND volume spike AND 1w uptrend
            if (close[i] > upper_bb[i] and 
                volume_spike_aligned[i] > 0.5 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower BB AND volume spike AND 1w downtrend
            elif (close[i] < lower_bb[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle BB
            if close[i] <= middle_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle BB
            if close[i] >= middle_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals