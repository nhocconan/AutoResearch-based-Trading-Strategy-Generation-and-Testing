#!/usr/bin/env python3
# 6h_RSI_Trend_Filter
# Hypothesis: RSI(14) with 6h timeframe, filtered by daily EMA50 trend, and volume confirmation.
# Uses RSI extremes (oversold <30 for long, overbought >70 for short) only when aligned with daily trend.
# Volume must be above 1.5x 20-period average to confirm momentum.
# Designed for 6h to achieve 50-150 total trades over 4 years, suitable for both bull and bear markets.

name = "6h_RSI_Trend_Filter"
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
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # RSI(14) on 6h close
    def rsi(arr, period=14):
        res = np.full_like(arr, np.nan)
        if len(arr) < period:
            return res
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        res = 100 - (100 / (1 + rs))
        res[:period] = np.nan
        return res
    
    rsi_14 = rsi(close, 14)
    
    # Align daily indicators to 6h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_14[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), price above daily EMA50, strong volume
            if rsi_14[i] < 30 and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), price below daily EMA50, strong volume
            elif rsi_14[i] > 70 and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 (momentum fading) or price below daily EMA50
            if rsi_14[i] > 50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 (momentum fading) or price above daily EMA50
            if rsi_14[i] < 50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals