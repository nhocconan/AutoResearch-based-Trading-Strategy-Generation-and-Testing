#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA200) and 1d volume spike confirmation.
# Long when price breaks above Donchian upper band AND weekly close > weekly EMA200 (bullish regime) AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower band AND weekly close < weekly EMA200 (bearish regime) AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel (mean of upper and lower bands).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 6h timeframe with strict entry conditions to avoid overtrading.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_Donchian20_Breakout_WeeklyEMA200_1dVolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate weekly EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)  # Volume > 2.0x 20-period MA
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian(20) channels on primary timeframe (6h)
    # Upper band: highest high over past 20 periods
    # Lower band: lowest low over past 20 periods
    # Middle band: (upper + lower) / 2 (exit level)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 00-23 UTC (6h bars less sensitive to session, but avoid quiet periods)
        hour = prices.index[i].hour
        in_session = (0 <= hour <= 23)  # Always in session for 6h
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND weekly close > weekly EMA200 (bullish regime) AND volume confirmation
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND weekly close < weekly EMA200 (bearish regime) AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian middle band (mean reversion)
            if close[i] <= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian middle band (mean reversion)
            if close[i] >= donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals