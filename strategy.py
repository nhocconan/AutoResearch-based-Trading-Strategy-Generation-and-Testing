#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and 4h volume confirmation (>1.5x 20-period average).
# Long when price breaks above Donchian upper band AND close > 1d EMA50 (bullish trend) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND close < 1d EMA50 (bearish trend) AND volume > 1.5x 20-period average.
# Exit when price retests the 20-period EMA (mean reversion) or opposite Donchian band touched.
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.5x) reduces false signals.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
# Donchian channels provide clear breakout levels effective in both bull and bear markets when combined with HTF trend filter.

name = "4h_Donchian20_Breakout_1dEMA50_4hVolumeConfirm_v1"
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
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA(20) for exit
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
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
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_20_4h[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND close > 1d EMA50 (bullish trend) AND volume confirm
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND close < 1d EMA50 (bearish trend) AND volume confirm
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests 4h EMA20 (mean reversion) OR touches Donchian lower (opposite band)
            if (close[i] <= ema_20_4h[i] or 
                close[i] < donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests 4h EMA20 (mean reversion) OR touches Donchian upper (opposite band)
            if (close[i] >= ema_20_4h[i] or 
                close[i] > donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals