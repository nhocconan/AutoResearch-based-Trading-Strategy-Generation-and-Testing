#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear power) with 1-day trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0, Bear Power < 0, price > 1d EMA34, volume > 1.5x average.
# Short when Bear Power > 0, Bull Power < 0, price < 1d EMA34, volume > 1.5x average.
# Designed for 6h to achieve 12-37 trades/year, works in both bull and bear markets.

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
    
    # 1d data for EMA34 trend and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Elder Ray components: Bull Power and Bear Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Align all indicators to lower timeframe (wait for 1d bar to close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > EMA34, strong volume
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_34_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0, price < EMA34, strong volume
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_34_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power > 0 or price < EMA34
            if bear_power[i] > 0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 or price > EMA34
            if bull_power[i] > 0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals