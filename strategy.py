#!/usr/bin/env python3
# 4h_RSI_Volume_Trend_Filter
# Hypothesis: RSI mean reversion with volume and 1d trend filter on 4h timeframe.
# RSI < 30 for long and > 70 for short, confirmed by volume spike and 1d EMA50 trend.
# Designed to avoid overtrading with strict conditions and work in both bull/bear markets via trend filter.

name = "4h_RSI_Volume_Trend_Filter"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # RSI(14) calculation
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # First average
        if len(close_prices) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average on 1d
    def mean_arr(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period - 1, len(arr)):
                res[i] = np.mean(arr[i - period + 1:i + 1])
        return res
    
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align 1d indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold, above 1d EMA50, volume spike
            if rsi[i] < 30 and close[i] > ema_50_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, below 1d EMA50, volume spike
            elif rsi[i] > 70 and close[i] < ema_50_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or price below EMA50
            if rsi[i] > 70 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or price above EMA50
            if rsi[i] < 30 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals