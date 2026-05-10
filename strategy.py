#!/usr/bin/env python3
# 12h_Donchian_Breakout_20_Trend_1dEMA34_Volume
# Hypothesis: Donchian(20) breakout on 12h with 1d EMA34 trend filter and volume confirmation. Works in bull markets via breakout continuation and in bear markets via short breakdowns. Volume filter reduces false breakouts. Target 15-35 trades/year.

name = "12h_Donchian_Breakout_20_Trend_1dEMA34_Volume"
timeframe = "12h"
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
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average on 1d
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for Donchian and indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for 20 periods
        highest_high = np.max(high[i-19:i+1])
        lowest_low = np.min(low[i-19:i+1])
        
        if position == 0:
            # Long: price breaks above upper Donchian, above EMA34, strong volume
            if close[i] > highest_high and close[i] > ema_34_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below EMA34, strong volume
            elif close[i] < lowest_low and close[i] < ema_34_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below lower Donchian or below EMA34
            if close[i] < lowest_low or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Donchian or above EMA34
            if close[i] > highest_high or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals