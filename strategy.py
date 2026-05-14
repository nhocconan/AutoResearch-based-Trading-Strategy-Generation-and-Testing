#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-period high with 1d ADX > 25 and 1d volume > 1.5x 20-period average.
# Short when price breaks below 20-period low with 1d ADX > 25 and 1d volume > 1.5x 20-period average.
# Exit on opposite Donchian level (20-period low for longs, 20-period high for shorts).
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 80-160 total trades over 4 years = 20-40/year for 4h timeframe.
# Works in bull/bear: 1d ADX ensures strong trend alignment, Donchian breakout captures momentum, 1d volume spike confirms institutional participation.

name = "4h_Donchian20_Breakout_1dADX_1dVolumeSpike"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[up_move < 0] = 0
    down_move[down_move < 0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14 = WilderSmooth(tr, 14)
    plus_dm_14 = WilderSmooth(up_move, 14)
    minus_dm_14 = WilderSmooth(down_move, 14)
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / (atr_14 + 1e-10)
    minus_di_14 = 100 * minus_dm_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = WilderSmooth(dx, 14)
    
    # 1d ADX > 25 trend filter
    adx_filter = adx_14 > 25
    
    # 1d volume confirmation: > 1.5x 20-period average (volume spike)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    
    # Align HTF indicators to 4h
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter.astype(float))
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or
            np.isnan(adx_filter_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + 1d ADX > 25 + 1d volume spike
            if (high[i] > high_roll[i-1] and 
                adx_filter_aligned[i] > 0.5 and
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + 1d ADX > 25 + 1d volume spike
            elif (low[i] < low_roll[i-1] and 
                  adx_filter_aligned[i] > 0.5 and
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low
            if low[i] < low_roll[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high
            if high[i] > high_roll[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals