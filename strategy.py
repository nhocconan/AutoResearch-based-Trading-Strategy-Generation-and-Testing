#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with 12h EMA50 uptrend and 4h volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian channel with 12h EMA50 downtrend and 4h volume > 1.5x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to limit fee churn and strict volume confirmation to reduce false breakouts.
# Target: 80-180 trades over 4 years (20-45/year) for 4h timeframe.
# Works in bull/bear: 12h EMA50 ensures trend alignment, Donchian provides structure within trend.

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_bullish = close_12h > ema_50  # Bullish if price above EMA50
    ema_50_bearish = close_12h < ema_50  # Bearish if price below EMA50
    
    # Align 12h indicators to 4h
    ema_50_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_50_bullish.astype(float))
    ema_50_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_50_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(ema_50_bullish_aligned[i]) or 
            np.isnan(ema_50_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + 12h uptrend + volume confirmation
            if (close[i] > high_rolling_max[i] and 
                ema_50_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + 12h downtrend + volume confirmation
            elif (close[i] < low_rolling_min[i] and 
                  ema_50_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < low_rolling_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > high_rolling_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals