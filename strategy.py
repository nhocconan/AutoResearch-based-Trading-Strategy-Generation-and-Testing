#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Donchian breakout captures momentum, 1w EMA50 filters for higher timeframe trend,
# volume confirmation ensures conviction. Long when price > upper band + EMA50 up + vol > 1.5x MA,
# Short when price < lower band + EMA50 down + vol > 1.5x MA. Exit on opposite band touch.
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels on 1d
    if len(high) >= 20:
        upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_prev = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().shift(1).values
    else:
        ema_50 = np.full(len(close_1w), np.nan)
        ema_50_prev = np.full(len(close_1w), np.nan)
    
    # Determine 1w EMA trend: up if current > previous
    ema_50_up = np.zeros(len(ema_50), dtype=bool)
    ema_50_down = np.zeros(len(ema_50), dtype=bool)
    for i in range(len(ema_50)):
        if not np.isnan(ema_50[i]) and not np.isnan(ema_50_prev[i]):
            ema_50_up[i] = ema_50[i] > ema_50_prev[i]
            ema_50_down[i] = ema_50[i] < ema_50_prev[i]
    
    # Align 1w EMA trend to 1d timeframe
    ema_50_up_aligned = align_htf_to_ltf(prices, df_1w, ema_50_up.astype(float))
    ema_50_down_aligned = align_htf_to_ltf(prices, df_1w, ema_50_down.astype(float))
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_up_aligned[i]) or np.isnan(ema_50_down_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > upper Donchian + 1w EMA up + volume filter
            if (close[i] > upper[i] and 
                ema_50_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < lower Donchian + 1w EMA down + volume filter
            elif (close[i] < lower[i] and 
                  ema_50_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian band
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches upper Donchian band
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals