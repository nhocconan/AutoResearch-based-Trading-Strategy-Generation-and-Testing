#!/usr/bin/env python3
# 4h_Trend_Momentum_Volume_Confirm
# Hypothesis: Combines 4h price momentum (close > EMA13) with 12h trend (close > EMA34) and volume confirmation (volume > 1.5x MA20) for long entries; opposite for shorts.
# Uses EMA13 for responsive momentum on 4h and EMA34 on 12h for trend filter to avoid counter-trend trades.
# Volume filter ensures breakouts have participation. Designed for 4h to achieve 20-50 trades/year.
# Works in bull markets via momentum and in bear via short signals when momentum fails.

name = "4h_Trend_Momentum_Volume_Confirm"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 4h EMA13 for momentum
    ema_13_4h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 12h EMA34 for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average on 12h
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_12h = mean_arr(volume_12h, 20)
    
    # Align all indicators to 4h timeframe (wait for 12h bar to close)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_13_4h[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h momentum up, 12h trend up, strong volume
            if close[i] > ema_13_4h[i] and close[i] > ema_34_12h_aligned[i] and volume[i] > 1.5 * vol_ma_20_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 4h momentum down, 12h trend down, strong volume
            elif close[i] < ema_13_4h[i] and close[i] < ema_34_12h_aligned[i] and volume[i] > 1.5 * vol_ma_20_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum breaks down or trend breaks down
            if close[i] < ema_13_4h[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum breaks up or trend breaks up
            if close[i] > ema_13_4h[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals