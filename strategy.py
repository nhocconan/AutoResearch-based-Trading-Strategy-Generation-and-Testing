#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_12hTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) on 6h with 12h EMA trend filter and volume confirmation. Bull Power > 0 and Bear Power < 0 indicate bullish/bearish momentum. 12h EMA ensures trend alignment. Volume confirms strength. Designed for 6h to achieve 50-150 total trades over 4 years (12-37/year). Works in bull/bear via trend filter.

name = "6h_ElderRay_BullBearPower_12hTrend_Volume"
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
    
    # 12h data for trend filter and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 12h volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_12h = mean_arr(volume_12h, 20)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar to close)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, above 12h EMA34, strong volume
            if bull_power[i] > 0 and close[i] > ema_34_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, below 12h EMA34, strong volume
            elif bear_power[i] < 0 and close[i] < ema_34_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or below 12h EMA34
            if bull_power[i] <= 0 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or above 12h EMA34
            if bear_power[i] >= 0 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals