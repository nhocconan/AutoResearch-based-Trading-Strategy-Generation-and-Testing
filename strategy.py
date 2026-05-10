#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) on 6h with daily trend filter (EMA34) and volume confirmation.
# Bull Power = High - EMA13 (bull strength), Bear Power = EMA13 - Low (bear strength).
# Long when Bull Power > 0 and rising, price > daily EMA34, volume > 2x average.
# Short when Bear Power > 0 and rising, price < daily EMA34, volume > 2x average.
# Designed for 6h to achieve 12-37 trades/year, works in both bull and bear markets by following daily trend.

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
    
    # Daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 6-day volume average for confirmation
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_6 = mean_arr(volume, 6)
    
    # Elder Ray components on 6h: Bull Power and Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Smooth Bull/Bear Power with 3-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Align daily indicators to 6h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_6_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_6_aligned[i]) or \
           np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power rising and positive, above daily EMA34, strong volume
            if bull_power_smooth[i] > 0 and bull_power_smooth[i] > bull_power_smooth[i-1] and \
               close[i] > ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_6_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power rising and positive, below daily EMA34, strong volume
            elif bear_power_smooth[i] > 0 and bear_power_smooth[i] > bear_power_smooth[i-1] and \
                 close[i] < ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_6_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or price drops below daily EMA34
            if bull_power_smooth[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative or price rises above daily EMA34
            if bear_power_smooth[i] <= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals