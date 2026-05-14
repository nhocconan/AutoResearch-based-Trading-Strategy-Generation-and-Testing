#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d trend filter.
# Long when price breaks above 20-period 6h high with 12h volume > 1.5x 20-period average and 1d EMA50 uptrend.
# Short when price breaks below 20-period 6h low with 12h volume > 1.5x 20-period average and 1d EMA50 downtrend.
# Exit on opposite Donchian level (20-period low for longs, high for shorts).
# Uses discrete position sizing (0.25) to limit fee churn and strict volume/confirmation to reduce false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
# Works in bull/bear: 1d EMA50 ensures trend alignment, Donchian provides structure, volume confirms breakout strength.

name = "6h_Donchian20_Breakout_12hVolumeConfirm_1dEMA50_Trend"
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
    
    # --- 6h Indicators (LTF) ---
    # 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h Volume for confirmation (will be overridden by 12h aligned volume)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- 12h Indicators (MTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_confirm = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)  # 20-period MA of 12h volume
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_bullish = close_1d > ema_50  # Bullish if price above EMA50
    ema_50_bearish = close_1d < ema_50  # Bearish if price below EMA50
    
    # Align 1d indicators to 6h
    ema_50_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_50_bullish.astype(float))
    ema_50_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_50_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema_50_bullish_aligned[i]) or 
            np.isnan(ema_50_bearish_aligned[i]) or
            np.isnan(vol_12h_confirm[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + 12h volume > 1.5x MA + 1d uptrend
            if (close[i] > high_roll[i] and 
                volume[i] > (1.5 * vol_12h_confirm[i]) and 
                ema_50_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + 12h volume > 1.5x MA + 1d downtrend
            elif (close[i] < low_roll[i] and 
                  volume[i] > (1.5 * vol_12h_confirm[i]) and 
                  ema_50_bearish_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals