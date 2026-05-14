#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike confirmation and 1w EMA50 trend filter.
# Long when price breaks above upper Donchian channel AND 1d volume > 2.0 * 20-period average AND 1w EMA50 is rising.
# Short when price breaks below lower Donchian channel AND 1d volume > 2.0 * 20-period average AND 1w EMA50 is falling.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 12h.

name = "12h_Donchian20_Breakout_1dVolumeConfirm_1wEMA50_Trend_v1"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_rising = ema_50_1w_aligned > np.roll(ema_50_1w_aligned, 1)
    ema_50_1w_rising[0] = False  # first value has no previous
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian channel (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_rising[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND volume confirmation AND EMA50 rising
            if (open_[i] <= highest_high[i] and close[i] > highest_high[i] and 
                volume_confirm_1d_aligned[i] > 0.5 and 
                ema_50_1w_rising[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND volume confirmation AND EMA50 falling
            elif (open_[i] >= lowest_low[i] and close[i] < lowest_low[i] and 
                  volume_confirm_1d_aligned[i] > 0.5 and 
                  not ema_50_1w_rising[i]):  # EMA50 falling
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