#!/usr/bin/env python3
# 4h_Keltner_Breakout_12hTrend_Volume
# Hypothesis: 4-hour breakouts from 20-period Keltner Channel with 12-hour EMA50 trend filter and volume confirmation.
# Keltner Channels (EMA-based ATR bands) adapt to volatility better than fixed bands. 12h EMA50 filters trend direction to avoid counter-trend trades.
# Volume confirmation (2x 20-period average) ensures breakout strength. Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Keltner_Breakout_12hTrend_Volume"
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
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period Keltner Channel: EMA20 ± 2*ATR(20)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2.0 * atr
    lower_keltner = ema_20 - 2.0 * atr
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_keltner[i]) or \
           np.isnan(lower_keltner[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner, above 12h EMA50, strong volume
            if close[i] > upper_keltner[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner, below 12h EMA50, strong volume
            elif close[i] < lower_keltner[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below EMA20 (middle of Keltner) or below 12h EMA50
            if close[i] < ema_20[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above EMA20 (middle of Keltner) or above 12h EMA50
            if close[i] > ema_20[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals