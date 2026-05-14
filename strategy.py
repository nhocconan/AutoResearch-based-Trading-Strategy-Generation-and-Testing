#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Uses Donchian channel (20-period high/low) from prior 1d for breakout structure,
# 1w EMA(34) for primary trend direction, and volume > 1.5x 20-bar average for conviction.
# Designed to capture strong breakouts in trending markets while avoiding false signals.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Targets 10-20 trades/year per symbol.

name = "1d_Donchian20_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Donchian channel (20) from prior bar
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: > 1.5x 20-bar average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # EMA(34) on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1w EMA(34)
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND weekly uptrend
            if close[i] > highest_20[i] and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND volume spike AND weekly downtrend
            elif close[i] < lowest_20[i] and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (breakdown) or weekly trend turns down
            if close[i] < lowest_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (breakout) or weekly trend turns up
            if close[i] > highest_20[i] or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals