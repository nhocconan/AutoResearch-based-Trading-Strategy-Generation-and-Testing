#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian(20) AND weekly EMA50 > EMA200 (bullish trend) AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below lower Donchian(20) AND weekly EMA50 < EMA200 (bearish trend) AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel (mean reversion within the channel).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 6h timeframe with strict entry conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h.

name = "6h_Donchian20_Breakout_1wEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian(20) channels (primary timeframe)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    midpoint = (upper + lower) / 2.0
    
    # Calculate weekly EMA50 and EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend = align_htf_to_ltf(prices, df_1w, ema_50 > ema_200)  # Boolean: True for bullish, False for bearish
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or
            np.isnan(ema_trend[i]) or np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND weekly EMA50 > EMA200 AND volume confirmation
            if (close[i-1] <= upper[i-1] and close[i] > upper[i] and 
                ema_trend[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND weekly EMA50 < EMA200 AND volume confirmation
            elif (close[i-1] >= lower[i-1] and close[i] < lower[i] and 
                  not ema_trend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals