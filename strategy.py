#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian channel AND 1d EMA34 > EMA200 (bullish trend) AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below lower Donchian channel AND 1d EMA34 < EMA200 (bearish trend) AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 50-150 total trades over 4 years (12-37/year) for 12h.
# Works in both bull and bear markets: 1d EMA crossover filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments.

name = "12h_Donchian20_Breakout_1dEMA34Trend_1dVolumeConfirm_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34 and EMA200 on 1d timeframe
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Bullish trend: EMA34 > EMA200, Bearish trend: EMA34 < EMA200
    ema_trend_bullish = ema34_1d > ema200_1d
    ema_trend_bearish = ema34_1d < ema200_1d
    
    # Calculate 1d volume confirmation filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align to 12h timeframe
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_bullish.astype(float))
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_trend_bearish.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_trend_bullish_aligned[i]) or 
            np.isnan(ema_trend_bearish_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian channel AND bullish 1d EMA trend AND volume confirmation
            if (low[i] <= highest_high[i-1] and high[i] > highest_high[i-1] and 
                ema_trend_bullish_aligned[i] > 0.5 and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian channel AND bearish 1d EMA trend AND volume confirmation
            elif (high[i] >= lowest_low[i-1] and low[i] < lowest_low[i-1] and 
                  ema_trend_bearish_aligned[i] > 0.5 and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if high[i] >= donchian_mid[i] and low[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if high[i] >= donchian_mid[i] and low[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals