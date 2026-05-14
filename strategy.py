#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and 12h volume confirmation (>1.5x 20-period average).
# Long when price breaks above Donchian upper (20-bar high) AND close > 1d EMA50 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below Donchian lower (20-bar low) AND close < 1d EMA50 (bearish trend) AND volume > 1.5x MA20.
# Exit when price crosses 1d EMA50 in opposite direction OR Donchian middle (10-bar average) is touched.
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 12h timeframe.
# Donchian channels provide clear structure; EMA50 filters for higher-timeframe trend; volume confirms conviction.

name = "12h_Donchian20_Breakout_1dEMA50_12hVolumeConfirm_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_20 + low_20) / 2.0
    # 12h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(donchian_middle[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: break above Donchian upper AND bullish trend (close > 1d EMA50) AND volume confirm
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below Donchian lower AND bearish trend (close < 1d EMA50) AND volume confirm
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 1d EMA50 OR touches Donchian middle (take profit)
            if (close[i] < ema_50_1d_aligned[i] or 
                close[i] <= donchian_middle[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above 1d EMA50 OR touches Donchian middle (take profit)
            if (close[i] > ema_50_1d_aligned[i] or 
                close[i] >= donchian_middle[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals