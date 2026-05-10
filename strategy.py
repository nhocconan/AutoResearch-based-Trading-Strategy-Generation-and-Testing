#!/usr/bin/env python3
# 12h_DailyTrend_Camarilla_R3_S3_Breakout_Volume
# Hypothesis: Daily trend filter (EMA34) filters out false breakouts, while 12h Camarilla R3/S3 levels provide precise entries.
# Volume confirmation ensures breakout strength. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear via mean-reversion at extreme levels when trend aligns.
# Target: 50-150 total trades over 4 years on 12h timeframe.

name = "12h_DailyTrend_Camarilla_R3_S3_Breakout_Volume"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA34 for trend (more stable than SMA)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 12h data for Camarilla pivot levels (based on previous 12h bar)
    # For 12h timeframe, we use 12h high/low/close to calculate pivots for current bar
    typical_price = (high + low + close) / 3
    range_hl = high - low
    # Camarilla R3 and S3 levels for current 12h bar
    R3 = typical_price + (range_hl * 1.1111)  # 1.1111 for R3
    S3 = typical_price - (range_hl * 1.1111)  # 1.1111 for S3
    
    # Volume confirmation (2-period average on 12h = ~1 day)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2) + 1  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3 with volume, above daily EMA34 (uptrend)
            if close[i] > R3[i] and volume_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below daily EMA34 (downtrend)
            elif close[i] < S3[i] and volume_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 or breaks below daily EMA34
            if close[i] < S3[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 or breaks above daily EMA34
            if close[i] > R3[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals