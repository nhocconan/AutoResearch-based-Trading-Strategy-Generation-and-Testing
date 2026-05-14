#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Uses Donchian channel breakout for trend capture, 1w EMA50 for robust trend filter, and 1d volume spike for momentum confirmation.
# Long when price breaks above Donchian upper channel AND 1w EMA50 is rising AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower channel AND 1w EMA50 is falling AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 30-100 total trades over 4 years (7-25/year) for 1d.

name = "1d_Donchian20_Breakout_1wEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "1d"
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
    
    # Calculate EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA50 slope (rising/falling)
    ema_50_slope_1w = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    ema_50_rising_1w = ema_50_slope_1w > 0
    ema_50_falling_1w = ema_50_slope_1w < 0
    
    # Align EMA50 trend to 1d timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising_1w.astype(float))
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling_1w.astype(float))
    
    # Calculate 1d volume confirmation filter
    if len(volume) < 20:
        return np.zeros(n)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Calculate Donchian channel (20-period)
    if len(high) < 20 or len(low) < 20:
        return np.zeros(n)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND EMA50 rising AND volume confirmation
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                ema_50_rising_aligned[i] > 0.5 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND EMA50 falling AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  ema_50_falling_aligned[i] > 0.5 and 
                  volume_confirm[i]):
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