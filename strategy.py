#!/usr/bin/env python3
# 12h_Three_Line_Breakout_1dTrend_Volume
# Hypothesis: 12-hour breakouts from daily 3-line breakout reversal levels with daily trend filter (EMA34) and volume confirmation.
# Daily EMA34 filters trend direction to avoid counter-trend trades; daily 3-line breakout levels provide precise entry/exit;
# Volume confirmation ensures breakout strength. Designed for 12h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "12h_Three_Line_Breakout_1dTrend_Volume"
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
    
    # Daily data for EMA34 trend filter and 3-line breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 3-line breakout levels (based on previous day)
    def calculate_3line(h, l, c):
        # 3-line breakout: uses high/low of previous day as support/resistance
        # Resistance = previous day's high
        # Support = previous day's low
        return h, l
    
    resistance = np.full_like(close_1d, np.nan)
    support = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        resistance[i], support[i] = calculate_3line(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance)
    support_aligned = align_htf_to_ltf(prices, df_1d, support)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(resistance_aligned[i]) or np.isnan(support_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above resistance, above daily EMA34, strong volume
            if close[i] > resistance_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below support, below daily EMA34, strong volume
            elif close[i] < support_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below support or below daily EMA34
            if close[i] < support_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above resistance or above daily EMA34
            if close[i] > resistance_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals