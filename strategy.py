#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above Donchian upper (20-period high) AND price > 1w EMA50 (bullish trend) AND volume > 2.0x 20-period average.
# Short when price breaks below Donchian lower (20-period low) AND price < 1w EMA50 (bearish trend) AND volume > 2.0x 20-period average.
# Exit on opposite Donchian breakout (price crosses below upper for longs, above lower for shorts).
# Uses 1w HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (2.0x) reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Donchian breakouts capture strong momentum moves, effective in both bull and bear markets when combined with HTF trend filter.

name = "1d_Donchian20_Breakout_1wEMA50_1dVolumeConfirm_v1"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 1d volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume > (2.0 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) - trend filter (smooth for 1d trading)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(volume_confirm_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND price > 1w EMA50 (bullish) AND volume confirm
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_confirm_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND price < 1w EMA50 (bearish) AND volume confirm
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_confirm_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Donchian upper (trend weakening)
            if close[i] < high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Donchian lower (trend weakening)
            if close[i] > low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals