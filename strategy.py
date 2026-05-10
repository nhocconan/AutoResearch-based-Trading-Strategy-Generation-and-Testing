#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray index (bull/bear power) combined with 1d EMA trend filter and volume confirmation.
# Bull power = high - EMA13, Bear power = EMA13 - low. Long when bull power > 0 and rising, price > 1d EMA50, volume > 1.5x average.
# Short when bear power > 0 and rising, price < 1d EMA50, volume > 1.5x average.
# Works in bull markets via bull power strength and in bear markets via bear power strength.
# Target: 12-37 trades/year on 6h timeframe.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray calculation
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema13 = ema(close, 13)
    
    # Bull power = high - EMA13, Bear power = EMA13 - low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d EMA50 for trend filter
    ema_50_1d = ema(close_1d, 50)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 1d EMA50 to 6h timeframe (wait for 1d bar to close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bull power positive and rising, price above 1d EMA50, strong volume
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power positive and rising, price below 1d EMA50, strong volume
            elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bull power turns negative or price drops below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bear power turns negative or price rises above 1d EMA50
            if bear_power[i] <= 0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals