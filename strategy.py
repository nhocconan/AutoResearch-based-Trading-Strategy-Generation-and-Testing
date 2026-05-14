#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and 1d volume spike confirmation.
# Uses 1w EMA34 for robust long-term trend detection (adapts to bull/bear regimes).
# Long when price breaks above Donchian upper band AND 1w EMA34 is rising AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 1w EMA34 is falling AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to limit fee churn. Target: 30-100 total trades over 4 years (7-25/year) for 1d.
# Works in both bull and bear markets: 1w EMA34 filter ensures we only trade with the long-term trend,
# while volume confirmation avoids breakouts in low-participation environments.

name = "1d_Donchian20_Breakout_1wEMA34_Trend_1dVolumeConfirm_v1"
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_rising_1w = np.zeros_like(close_1w, dtype=bool)
    ema_rising_1w[1:] = ema_34_1w[1:] > ema_34_1w[:-1]
    ema_rising_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_rising_1w.astype(float))
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian(20) channels (based on prior 20 periods)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback:
            continue
        window_start = i - lookback
        window_end = i  # exclude current bar (use prior completed periods)
        highest_high[i] = np.max(high[window_start:window_end])
        lowest_low[i] = np.min(low[window_start:window_end])
        donchian_mid[i] = (highest_high[i] + lowest_low[i]) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_rising_1w_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND 1w EMA34 rising AND volume confirmation
            if (open_[i] <= highest_high[i] and close[i] > highest_high[i] and 
                ema_rising_1w_aligned[i] > 0.5 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND 1w EMA34 falling AND volume confirmation
            elif (open_[i] >= lowest_low[i] and close[i] < lowest_low[i] and 
                  ema_rising_1w_aligned[i] < 0.5 and 
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