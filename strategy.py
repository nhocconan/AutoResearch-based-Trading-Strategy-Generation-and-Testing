#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation. 
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-period average volume. 
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-period average volume. 
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing strong trends with volume confirmation while avoiding false breakouts in low-volume environments.

name = "1d_DonchianBreakout_VolumeTrend_1wEMA50_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    for i in range(n):
        if i < lookback - 1:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
    
    # Calculate 20-period average volume for confirmation
    vol_lookback = 20
    avg_volume = np.zeros(n)
    
    for i in range(n):
        if i < vol_lookback - 1:
            avg_volume[i] = np.nan
        else:
            start_idx = i - vol_lookback + 1
            avg_volume[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND price > 1w EMA50 AND volume > 1.5x average volume
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low AND price < 1w EMA50 AND volume > 1.5x average volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low (reverse signal) OR price < 1w EMA50 (trend filter fail)
            if (close[i] < lowest_low[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high (reverse signal) OR price > 1w EMA50 (trend filter fail)
            if (close[i] > highest_high[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals