#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian breakout (20) for direction and 1d EMA50 trend filter, with 1h volume confirmation (>1.5x 20-period average) for entry timing.
# Long when price breaks above 4h Donchian upper channel AND close > 1d EMA50 AND volume > 1.5x MA20.
# Short when price breaks below 4h Donchian lower channel AND close < 1d EMA50 AND volume > 1.5x MA20.
# Exit when price crosses the 4h Donchian midline (average of upper/lower) OR volume drops below 1.2x MA20.
# Uses 4h for structure (Donchian channels), 1d for higher-timeframe trend filter (EMA50), and 1h only for precise entry timing and volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to stay within fee drag limits for 1h timeframe.
# Donchian breakouts capture sustained moves, EMA50 filter avoids counter-trend trades, volume confirmation reduces false breakouts.

name = "1h_Donchian20_Breakout_4hDir_1dEMA50_1hVolumeConfirm_v1"
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
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1h = volume > (1.5 * vol_ma_20)
    volume_exit = volume < (1.2 * vol_ma_20)  # Exit on low volume
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian Channel (20)
    donchian_window = 20
    upper_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle_4h = (upper_4h + lower_4h) / 2.0  # Midline for exit
    
    # Align 4h indicators to 1h timeframe (wait for completed 4h bar)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or np.isnan(middle_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm_1h[i]) or np.isnan(volume_exit[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4h Donchian upper AND close > 1d EMA50 AND volume confirm
            if (close[i] > upper_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_1h[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h Donchian lower AND close < 1d EMA50 AND volume confirm
            elif (close[i] < lower_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_1h[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 4h Donchian midline OR low volume exit
            if (close[i] < middle_4h_aligned[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above 4h Donchian midline OR low volume exit
            if (close[i] > middle_4h_aligned[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals