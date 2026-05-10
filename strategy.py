#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: 1-hour breakouts from 4-hour Camarilla R1/S1 levels with daily trend filter (EMA34) and volume confirmation.
# 4h trend filter avoids counter-trend trades; 4h Camarilla levels provide precise entry/exit; daily volume confirms strength.
# Designed for 1h to achieve 15-37 trades/year, suitable for both bull and bear markets by filtering with higher timeframes.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 4h Camarilla levels (based on previous 4h bar)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R1 = c + (range_ * 1.1000 / 12)
        S1 = c - (range_ * 1.1000 / 12)
        return R1, S1
    
    R1_4h = np.full_like(close_4h, np.nan)
    S1_4h = np.full_like(close_4h, np.nan)
    for i in range(1, len(close_4h)):
        R1_4h[i], S1_4h[i] = calculate_camarilla(high_4h[i-1], low_4h[i-1], close_4h[i-1])
    
    # 1d volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align 4h indicators to 1h timeframe (wait for 4h bar to close)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Align 1d volume to 1h timeframe (wait for 1d bar to close)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or \
           np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA34, strong daily volume
            if close[i] > R1_4h_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, below 4h EMA34, strong daily volume
            elif close[i] < S1_4h_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > 2.0 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below 4h EMA34
            if close[i] < S1_4h_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R1 or above 4h EMA34
            if close[i] > R1_4h_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals