#!/usr/bin/env python3
# Hypothesis: 1h timeframe with 4h trend filter and volume confirmation. Uses 4h EMA for trend direction (bullish when close > EMA20, bearish when close < EMA20) and enters on 1h pullbacks to EMA10 with volume above 1.5x average. This captures trend continuation in both bull and bear markets while limiting trades via volume filter. Target: 60-150 total trades over 4 years (15-37/year) with size 0.20.

name = "1h_4hEMA_Trend_VolumePullback"
timeframe = "1h"
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
    
    # 4h EMA20 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close']
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1h EMA10 for pullback entry
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).values
    
    # Volume filter: 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean()
    vol_threshold = 1.5 * vol_ma_20.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_10[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h bullish trend + price pulls back to 1h EMA10 + volume spike
            if close[i] > ema_20_4h_aligned[i] and close[i] <= ema_10[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h bearish trend + price bounces to 1h EMA10 + volume spike
            elif close[i] < ema_20_4h_aligned[i] and close[i] >= ema_10[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or price moves above EMA10
            if close[i] < ema_20_4h_aligned[i] or close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal or price moves below EMA10
            if close[i] > ema_20_4h_aligned[i] or close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals