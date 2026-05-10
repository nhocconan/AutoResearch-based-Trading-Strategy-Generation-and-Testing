#!/usr/bin/env python3
# 1d_RSI_Trend_Filter_Volume_Spike
# Hypothesis: Daily RSI extreme with weekly EMA trend filter and volume spike. 
# RSI < 30 for long, > 70 for short in trending markets (EMA50 direction). 
# Volume > 2x 20-day average confirms momentum. Designed for low-frequency, high-conviction trades.
# Weekly trend filter avoids counter-trend trades in choppy markets.

name = "1d_RSI_Trend_Filter_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14)
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period - 1, len(arr)):
                res[i] = np.mean(arr[i - period + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align weekly indicators to daily timeframe (wait for weekly bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold, above weekly EMA50 (uptrend), volume spike
            if rsi[i] < 30 and close[i] > ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, below weekly EMA50 (downtrend), volume spike
            elif rsi[i] > 70 and close[i] < ema_50_1w_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend change
            if rsi[i] > 70 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or trend change
            if rsi[i] < 30 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals