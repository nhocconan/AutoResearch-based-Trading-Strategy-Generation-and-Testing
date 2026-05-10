#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Daily breakouts from weekly Camarilla R1/S1 levels with weekly EMA34 trend filter and volume confirmation.
# Weekly EMA34 filters trend direction to avoid counter-trend trades; weekly Camarilla levels provide precise entry/exit;
# Volume confirmation ensures breakout strength. Designed for 1d to achieve 7-25 trades/year, suitable for both bull and bear markets.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Weekly data for EMA34 trend filter and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Camarilla levels (based on previous week)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R1 = c + (range_ * 1.1000 / 12)
        S1 = c - (range_ * 1.1000 / 12)
        return R1, S1
    
    R1 = np.full_like(close_1w, np.nan)
    S1 = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        R1[i], S1[i] = calculate_camarilla(high_1w[i-1], low_1w[i-1], close_1w[i-1])
    
    # Weekly volume confirmation: 10-period average (weekly data has fewer bars)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_10 = mean_arr(volume_1w, 10)
    
    # Align weekly indicators to daily timeframe (wait for weekly bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    vol_ma_10_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_10_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above weekly EMA34, strong volume
            if close[i] > R1_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > 2.0 * vol_ma_10_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below weekly EMA34, strong volume
            elif close[i] < S1_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > 2.0 * vol_ma_10_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below weekly EMA34
            if close[i] < S1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or above weekly EMA34
            if close[i] > R1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals