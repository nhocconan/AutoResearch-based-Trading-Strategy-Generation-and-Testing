#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper channel AND 1d EMA34 is rising AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower channel AND 1d EMA34 is falling AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 75-200 total trades over 4 years (19-50/year) for 4h.

name = "4h_Donchian20_Breakout_1dEMA34_Trend_1dVolumeConfirm_v1"
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
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    high_roll = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_roll = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_prev = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().shift(1).values
    ema_34_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_prev)
    ema_34_rising = ema_34_1d_aligned > ema_34_1d_prev_aligned
    ema_34_falling = ema_34_1d_aligned < ema_34_1d_prev_aligned
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_period, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper channel AND EMA34 rising AND volume confirmation
            if (open_[i] <= high_roll[i] and close[i] > high_roll[i] and 
                ema_34_rising[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower channel AND EMA34 falling AND volume confirmation
            elif (open_[i] >= low_roll[i] and close[i] < low_roll[i] and 
                  ema_34_falling[i] and 
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