#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend_12h
# Hypothesis: Donchian channel breakouts with volume confirmation and 12h trend filter capture strong momentum moves while avoiding false breakouts in choppy markets.
# The 12h EMA filter ensures we only trade in the direction of the higher timeframe trend, improving win rate.
# Volume confirmation ensures breakouts are supported by participation.
# Designed for low trade frequency (20-40/year) to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=donch_period, min_periods=donch_period).max()
    donch_lower = low_series.rolling(window=donch_period, min_periods=donch_period).min()
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    # 12h EMA trend filter (trained on 12h data, aligned to 4h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above upper Donchian AND volume confirmation AND 12h uptrend
            if close[i] > donch_upper[i] and vol_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower Donchian AND volume confirmation AND 12h downtrend
            elif close[i] < donch_lower[i] and vol_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian OR 12h trend turns down
            if close[i] < donch_lower[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian OR 12h trend turns up
            if close[i] > donch_upper[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals