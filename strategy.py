#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian(20) AND 1d EMA50 is rising AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below lower Donchian(20) AND 1d EMA50 is falling AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 12h.
# Works in both bull and bear markets: Donchian breakouts capture strong trends, EMA50 filter avoids counter-trend trades,
# volume confirmation ensures breakout validity, and midpoint exit provides symmetric risk/reward.

name = "12h_Donchian20_Breakout_1dEMA50_Trend_1dVolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA50 slope (rising/falling)
    ema_50_slope_1d = np.diff(ema_50_1d, prepend=ema_50_1d[0])
    ema_50_rising_1d = ema_50_slope_1d > 0
    ema_50_falling_1d = ema_50_slope_1d < 0
    
    # Calculate 1d volume confirmation filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align HTF indicators to 12h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising_1d.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling_1d.astype(float))
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian(20) channels on 12h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND EMA50 rising AND volume confirmation
            if (open_[i] <= highest_high_20[i] and close[i] > highest_high_20[i] and 
                ema_50_rising_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND EMA50 falling AND volume confirmation
            elif (open_[i] >= lowest_low_20[i] and close[i] < lowest_low_20[i] and 
                  ema_50_falling_aligned[i] > 0.5 and 
                  volume_confirm_1d_aligned[i] > 0.5):
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