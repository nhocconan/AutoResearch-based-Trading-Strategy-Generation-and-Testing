#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 4h volume confirmation (>2.0x 20-period average).
# Long when price breaks above Donchian upper AND close > 12h EMA50 AND volume > 2.0x 20-period average.
# Short when price breaks below Donchian lower AND close < 12h EMA50 AND volume > 2.0x 20-period average.
# Exit when price retraces to 12h EMA50 (mean reversion to trend).
# Uses 12h HTF for trend to reduce noise and overtrading. Volume confirmation (2.0x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Donchian breakouts capture momentum; EMA50 filter ensures alignment with intermediate trend.

name = "4h_Donchian20_Breakout_12hEMA50_4hVolumeConfirm_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # 4h Donchian Channel (20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) - trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND close > 12h EMA50 AND volume confirm
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND close < 12h EMA50 AND volume confirm
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retraces to 12h EMA50 (mean reversion to trend)
            if close[i] <= ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retraces to 12h EMA50 (mean reversion to trend)
            if close[i] >= ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals