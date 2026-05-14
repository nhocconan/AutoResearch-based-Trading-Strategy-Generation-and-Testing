#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and 12h volume confirmation.
# Long when price breaks above upper Donchian channel AND 1d EMA34 is rising AND 12h volume > 2.0 * 20-period average volume.
# Short when price breaks below lower Donchian channel AND 1d EMA34 is falling AND 12h volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 12h.
# Works in both bull and bear markets: 1d EMA34 filter ensures we only trade with the long-term trend,
# while volume confirmation avoids breakouts in low-participation environments.

name = "12h_Donchian20_Breakout_1dEMA34_Trend_12hVolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_rising_1d = np.zeros_like(close_1d, dtype=bool)
    ema_rising_1d[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_rising_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_rising_1d.astype(float))
    
    # Calculate 12h volume confirmation filter (primary TF)
    vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (2.0 * vol_ma_20_12h)
    
    # Calculate Donchian channel (20-period) on primary TF
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_rising_1d_aligned[i]) or 
            np.isnan(volume_confirm_12h[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND 1d EMA34 rising AND volume confirmation
            if (open_[i] <= highest_high_20[i] and close[i] > highest_high_20[i] and 
                ema_rising_1d_aligned[i] > 0.5 and 
                volume_confirm_12h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND 1d EMA34 falling AND volume confirmation
            elif (open_[i] >= lowest_low_20[i] and close[i] < lowest_low_20[i] and 
                  ema_rising_1d_aligned[i] < 0.5 and 
                  volume_confirm_12h[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals