#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > 1d EMA50) and volume confirmation (>1.5x 20-period average).
# Long when price breaks above Donchian upper channel AND close > 1d EMA50 AND volume > 1.5x MA20.
# Short when price breaks below Donchian lower channel AND close < 1d EMA50 AND volume > 1.5x MA20.
# Exit when price reverses to touch Donchian midpoint (mean reversion within channel) OR trend filter fails.
# Uses 1d HTF for trend to avoid counter-trend trades. Volume confirmation reduces false breakouts.
# Target: 80-180 total trades over 4 years (20-45/year) to stay within fee drag limits for 6h timeframe.
# Donchian channels provide clear breakout levels; EMA50 filter ensures alignment with higher timeframe trend.

name = "6h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Donchian Channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    # Volume confirmation: > 1.5x 20-period average (reduces false signals)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above upper channel AND bullish trend AND volume confirm
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower channel AND bearish trend AND volume confirm
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches midpoint (mean reversion) OR trend fails
            if (close[i] <= donchian_mid[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches midpoint (mean reversion) OR trend fails
            if (close[i] >= donchian_mid[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals