#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and 12h volume confirmation (>1.5x 20-period average).
# Long when price breaks above Donchian(20) high AND close > 1d EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below Donchian(20) low AND close < 1d EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price crosses 1d EMA34 in opposite direction (trend change).
# Donchian channels provide clear structure, EMA34 filters for primary trend, volume confirmation reduces false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Works in both bull and bear markets by only taking trades in direction of higher timeframe trend.

name = "12h_Donchian20_Breakout_1dEMA34_12hVolumeConfirm_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 12h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND close > 1d EMA34 AND volume confirm
            if (high[i] > high_20[i-1] and  # breakout above previous period's high
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND close < 1d EMA34 AND volume confirm
            elif (low[i] < low_20[i-1] and  # breakout below previous period's low
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA34 (trend change to bearish)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA34 (trend change to bullish)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals