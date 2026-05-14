#!/usr/bin/env python3
# Hypothesis: 1h Donchian channel breakout with 4h EMA50 trend filter and 1d volume confirmation (>1.5x 20-period average).
# Long when price breaks above Donchian(20) high AND close > 4h EMA50 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below Donchian(20) low AND close < 4h EMA50 (bearish trend) AND volume > 1.5x MA20.
# Exit when price crosses 4h EMA50 in opposite direction.
# Uses 4h HTF for trend to reduce noise and overtrading. Volume confirmation reduces false signals.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Donchian breakouts capture momentum; EMA50 trend filter avoids counter-trend trades; volume confirmation ensures conviction.

name = "1h_Donchian20_Breakout_4hEMA50_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 1h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1h = volume > (1.5 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) - trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_confirm_1h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND close > 4h EMA50 AND volume confirm
            if (close[i] > high_20[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_confirm_1h[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below Donchian low AND close < 4h EMA50 AND volume confirm
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_confirm_1h[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 4h EMA50 (trend change)
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price crosses above 4h EMA50 (trend change)
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals