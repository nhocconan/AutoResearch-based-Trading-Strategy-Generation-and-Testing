#!/usr/bin/env python3
# 6h_Relative_Strength_Index_with_Trend_and_Volume
# Hypothesis: Combines RSI momentum with trend direction (1d EMA) and volume confirmation.
# In bull markets: buy when RSI < 30 (oversold) but price > 1d EMA200 (uptrend).
# In bear markets: sell when RSI > 70 (overbought) but price < 1d EMA200 (downtrend).
# Uses volume filter to avoid low-liquidity false signals. Targets 15-30 trades/year.

name = "6h_RSI_Trend_Volume_Filter"
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
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 24-period average (4 days)
    def sma(arr, window):
        res = np.full_like(arr, np.nan)
        if len(arr) >= window:
            for i in range(window-1, len(arr)):
                res[i] = np.mean(arr[i-window+1:i+1])
        return res
    vol_ma = sma(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14, 24)
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: RSI oversold (<30) + uptrend (price > EMA200) + volume
            if rsi[i] < 30 and close[i] > ema_200_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + downtrend (price < EMA200) + volume
            elif rsi[i] > 70 and close[i] < ema_200_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought OR trend breaks
            if rsi[i] > 70 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold OR trend breaks
            if rsi[i] < 30 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals