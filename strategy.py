#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and 12h volume spike (>2.0x 20-period average).
# Long when price breaks above Donchian(20) upper band AND close > 1d EMA34 (bullish trend) AND volume > 2.0x MA20.
# Short when price breaks below Donchian(20) lower band AND close < 1d EMA34 (bearish trend) AND volume > 2.0x MA20.
# Exit when price crosses 1d EMA34 in opposite direction (trend change).
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>2.0x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Donchian channels provide clear structure; EMA34 filter ensures alignment with higher timeframe trend.

name = "12h_Donchian20_Breakout_1dEMA34_12hVolumeSpike_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 12h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (2.0 * vol_ma_20)
    
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
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND close > 1d EMA34 (bullish trend) AND volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band AND close < 1d EMA34 (bearish trend) AND volume spike
            elif (close[i] < lowest_20[i] and 
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