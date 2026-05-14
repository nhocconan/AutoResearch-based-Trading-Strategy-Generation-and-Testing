#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50 slope) and volume confirmation.
# Long when price breaks above Donchian upper band with 1d EMA50 sloping up and 6h volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band with 1d EMA50 sloping down and 6h volume > 1.5x 20-period average.
# Exit on opposite Donchian band (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. Donchian provides structure, 1d EMA50 slope confirms
# higher timeframe trend, volume confirms momentum. Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Works in bull/bear: 1d EMA50 slope adapts to trend, Donchian breakouts capture momentum, volume filters weak moves.

name = "6h_Donchian20_Breakout_1dEMA50Slope_6hVolumeConfirm"
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
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h Donchian(20) - highest high and lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 1d EMA50 slope: positive if current > 5 periods ago
    ema_50_slope = np.zeros_like(ema_50)
    ema_50_slope[5:] = ema_50[5:] > ema_50[:-5]
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_confirm_6h[i]) or
            np.isnan(ema_50_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 1d EMA50 sloping up + 6h volume confirmation
            if (close[i] > highest_high[i] and 
                ema_50_slope_aligned[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 1d EMA50 sloping down + 6h volume confirmation
            elif (close[i] < lowest_low[i] and 
                  not ema_50_slope_aligned[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals