#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d trend filter (EMA50).
# Uses Donchian channel breakouts for momentum capture, confirmed by HTF volume spike and 1d EMA50 trend direction.
# Long when price breaks above upper Donchian band AND 12h volume > 1.5 * 20-period average AND close > 1d EMA50.
# Short when price breaks below lower Donchian band AND 12h volume > 1.5 * 20-period average AND close < 1d EMA50.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 80-180 total trades over 4 years (20-45/year) for 4h.

name = "4h_Donchian20_Breakout_12hVolumeConfirm_1dEMA50Trend_v1"
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
    
    # Calculate Donchian Channel (20-period)
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    mid_band = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            window_high = high[i - lookback + 1:i + 1]
            window_low = low[i - lookback + 1:i + 1]
            upper_band[i] = np.max(window_high)
            lower_band[i] = np.min(window_low)
            mid_band[i] = (upper_band[i] + lower_band[i]) / 2
    
    # Calculate 12h volume confirmation filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume_12h > (1.5 * vol_ma_20_12h)
    volume_confirm_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm_12h.astype(float))
    
    # Calculate 1d EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(mid_band[i]) or
            np.isnan(volume_confirm_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper band AND volume confirmation AND 1d EMA50 uptrend (close > EMA50)
            if (open_[i] <= upper_band[i] and close[i] > upper_band[i] and 
                volume_confirm_12h_aligned[i] > 0.5 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower band AND volume confirmation AND 1d EMA50 downtrend (close < EMA50)
            elif (open_[i] >= lower_band[i] and close[i] < lower_band[i] and 
                  volume_confirm_12h_aligned[i] > 0.5 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to midpoint of Donchian channel
            if close[i] <= mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to midpoint of Donchian channel
            if close[i] >= mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals