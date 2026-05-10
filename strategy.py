#!/usr/bin/env python3
# 4h_HTF1d_RSI_CCI_Reversal
# Hypothesis: 4-hour reversals triggered by daily RSI(14) and CCI(20) extremes. Uses daily RSI < 30 for long and > 70 for short, confirmed by CCI crossing zero and volume spike. Works in both bull and bear markets by capturing mean reversion at extremes with trend filter via CCI zero-cross. Designed for ~30-50 trades/year.

name = "4h_HTF1d_RSI_CCI_Reversal"
timeframe = "4h"
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
    
    # Daily data for RSI and CCI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily RSI(14)
    def rsi(arr, period):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        rsi_full = np.full_like(arr, np.nan)
        rsi_full[period:] = rsi_val[period:]
        return rsi_full
    
    rsi_14 = rsi(close_1d, 14)
    
    # Daily CCI(20)
    def cci(high, low, close, period):
        tp = (high + low + close) / 3.0
        sma = np.zeros_like(tp)
        mean_dev = np.zeros_like(tp)
        for i in range(period-1, len(tp)):
            sma[i] = np.mean(tp[i-period+1:i+1])
            mean_dev[i] = np.mean(np.abs(tp[i-period+1:i+1] - sma[i]))
        cci_val = np.where(mean_dev != 0, (tp - sma) / (0.015 * mean_dev), 0)
        cci_full = np.full_like(tp, np.nan)
        cci_full[period-1:] = cci_val[period-1:]
        return cci_full
    
    cci_20 = cci(high_1d, low_1d, close_1d, 20)
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 4h timeframe (wait for 1d bar to close)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    cci_20_aligned = align_htf_to_ltf(prices, df_1d, cci_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_14_aligned[i]) or np.isnan(cci_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), CCI crosses above zero, volume spike
            if rsi_14_aligned[i] < 30 and cci_20_aligned[i] > 0 and cci_20_aligned[i-1] <= 0 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), CCI crosses below zero, volume spike
            elif rsi_14_aligned[i] > 70 and cci_20_aligned[i] < 0 and cci_20_aligned[i-1] >= 0 and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 50 or CCI < -100
            if rsi_14_aligned[i] > 50 or cci_20_aligned[i] < -100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 or CCI > 100
            if rsi_14_aligned[i] < 50 or cci_20_aligned[i] > 100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals