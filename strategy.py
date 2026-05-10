#!/usr/bin/env python3
# 6h_LongTermTrend_SwingReversion
# Hypothesis: On 6-hour timeframe, trade reversions to the long-term trend (12h EMA50) after pullbacks, with volume confirmation and momentum filter (6h RSI). 
# Uses 12h EMA50 as trend filter, 6h RSI for overbought/oversold conditions, and volume spike for entry confirmation. Designed to work in both bull and bear markets by following the higher timeframe trend.

name = "6h_LongTermTrend_SwingReversion"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 6h RSI for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 12h EMA50 to 6h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to 12h EMA50 from above, RSI oversold, volume spike
            if close[i] <= ema_50_12h_aligned[i] * 1.005 and close[i] >= ema_50_12h_aligned[i] * 0.995 and \
               rsi[i] < 30 and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to 12h EMA50 from below, RSI overbought, volume spike
            elif close[i] >= ema_50_12h_aligned[i] * 0.995 and close[i] <= ema_50_12h_aligned[i] * 1.005 and \
                 rsi[i] > 70 and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves away from 12h EMA50 or RSI overbought
            if close[i] > ema_50_12h_aligned[i] * 1.02 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves away from 12h EMA50 or RSI oversold
            if close[i] < ema_50_12h_aligned[i] * 0.98 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals