#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and 1d volume confirmation.
# Long when price breaks above upper Donchian channel AND 1w EMA50 > EMA200 (bullish trend) AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below lower Donchian channel AND 1w EMA50 < EMA200 (bearish trend) AND 1d volume > 1.5 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 30-100 total trades over 4 years (7-25/year) for 1d.
# Works in both bull and bear markets: 1w EMA crossover filter ensures we only trade in clear trending conditions,
# while volume confirmation avoids breakouts in low-participation environments.

name = "1d_Donchian20_Breakout_1wEMATrend_1dVolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 and EMA200 on 1w timeframe
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Bullish trend: EMA50 > EMA200, Bearish trend: EMA50 < EMA200
    ema_trend_bullish = ema50_1w > ema200_1w
    ema_trend_bearish = ema50_1w < ema200_1w
    
    # Align to 1d timeframe
    ema_trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_bullish.astype(float))
    ema_trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_bearish.astype(float))
    
    # Calculate 1d volume confirmation filter (primary TF)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate Donchian channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_trend_bullish_aligned[i]) or 
            np.isnan(ema_trend_bearish_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND bullish 1w EMA trend AND volume confirmation
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                ema_trend_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND bearish 1w EMA trend AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  ema_trend_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
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