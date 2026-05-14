#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average).
# Long when price breaks above upper Donchian(20) AND close > 1d EMA34 (bullish trend) AND volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20) AND close < 1d EMA34 (bearish trend) AND volume > 1.5x 20-period average.
# Exit when price retests the 1d EMA34 level (mean reversion to trend) or opposite Donchian level touched.
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.5x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe.
# Donchian channels provide clear structure, effective in both bull and bear markets when combined with HTF trend filter.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v2"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20)
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian AND close > 1d EMA34 (bullish trend) AND volume confirm
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian AND close < 1d EMA34 (bearish trend) AND volume confirm
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests 1d EMA34 (mean reversion to trend) OR touches lower Donchian (opposite level)
            if (close[i] <= ema_34_1d_aligned[i] or 
                close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests 1d EMA34 (mean reversion to trend) OR touches upper Donchian (opposite level)
            if (close[i] >= ema_34_1d_aligned[i] or 
                close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals