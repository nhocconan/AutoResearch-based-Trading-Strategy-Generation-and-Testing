#!/usr/bin/env python3
# 6h_Momentum_Fade_With_Volume_And_Trend_Filter
# Hypothesis: Fade extreme moves using 6h RSI extremes combined with 1d trend filter and volume confirmation.
# In strong trends (1d EMA50), look for overextended RSI readings for mean reversion entries.
# Works in bull markets by fading overextended rallies, in bear markets by fading oversold bounces.
# Uses discrete position sizing (0.25) to limit turnover. Target: 20-50 trades/year.

name = "6h_Momentum_Fade_With_Volume_And_Trend_Filter"
timeframe = "6h"
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
    
    # 6h RSI (14-period) for mean reversion signals
    def rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        ma_up = np.zeros_like(arr)
        ma_down = np.zeros_like(arr)
        # Wilder's smoothing
        ma_up[period-1] = np.mean(up[:period])
        ma_down[period-1] = np.mean(down[:period])
        for i in range(period, len(arr)):
            ma_up[i] = (ma_up[i-1] * (period-1) + up[i]) / period
            ma_down[i] = (ma_down[i-1] * (period-1) + down[i]) / period
        rs = np.where(ma_down != 0, ma_up / ma_down, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_6h = rsi(close, 14)
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume confirmation: current volume > 1.5x 20-period average
    def mean_arr(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period - 1, len(arr)):
                res[i] = np.mean(arr[i - period + 1:i + 1])
        return res
    
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_6h[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend determination: price above/below 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume condition
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: RSI oversold (<30) in uptrend with volume confirmation
            if is_uptrend and rsi_6h[i] < 30 and volume_condition:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) in downtrend with volume confirmation
            elif is_downtrend and rsi_6h[i] > 70 and volume_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or trend turns down
            if rsi_6h[i] >= 50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or trend turns up
            if rsi_6h[i] <= 50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals