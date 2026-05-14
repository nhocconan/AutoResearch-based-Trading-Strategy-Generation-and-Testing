#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter.
# Long when price breaks above upper Donchian(20) with 1d volume > 2.0x 20-period average and 1w EMA34 uptrend.
# Short when price breaks below lower Donchian(20) with 1d volume > 2.0x 20-period average and 1w EMA34 downtrend.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Works in bull/bear: 1w EMA34 ensures strong trend alignment, Donchian provides clear structure, 1d volume spike confirms institutional participation.

name = "4h_Donchian20_Breakout_1dVolumeSpike_1wEMA34_Trend"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 2.0x 20-period average (volume spike)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    
    # Align 1d volume confirmation to 4h
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA34 trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w EMA34 uptrend/downtrend signals
    ema_34_uptrend = ema_34_1w_aligned > np.roll(ema_34_1w_aligned, 1)
    ema_34_downtrend = ema_34_1w_aligned < np.roll(ema_34_1w_aligned, 1)
    # Handle first value
    ema_34_uptrend[0] = False
    ema_34_downtrend[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(ema_34_uptrend[i]) or
            np.isnan(ema_34_downtrend[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + 1d volume spike + 1w EMA34 uptrend
            if (high[i] > highest_20[i] and 
                volume_confirm_1d_aligned[i] > 0.5 and
                ema_34_uptrend[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + 1d volume spike + 1w EMA34 downtrend
            elif (low[i] < lowest_20[i] and 
                  volume_confirm_1d_aligned[i] > 0.5 and
                  ema_34_downtrend[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if low[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if high[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals