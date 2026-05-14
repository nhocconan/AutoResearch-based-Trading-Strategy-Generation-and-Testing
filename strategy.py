#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and 4h volume confirmation.
# Long when price breaks above Donchian upper (20-period high) AND price > 1d EMA34 (bullish trend) AND 4h volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower (20-period low) AND price < 1d EMA34 (bearish trend) AND 4h volume > 1.5x 20-period average.
# Exit when price crosses the 10-period EMA in opposite direction (trailing exit).
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.5x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Donchian breakouts capture momentum, effective in both bull and bear markets when combined with HTF trend filter.

name = "4h_Donchian20_Breakout_1dEMA34_4hVolumeConfirm_v1"
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
    # 4h Donchian(20) - price channel breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 4h EMA(10) for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    # 4h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter (smooth for 4h trading)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(ema_10[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND price > 1d EMA34 (bullish) AND 4h volume confirm
            if (close[i] > high_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND price < 1d EMA34 (bearish) AND 4h volume confirm
            elif (close[i] < low_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 10-period EMA (momentum weakening)
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above 10-period EMA (momentum weakening)
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals