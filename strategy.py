#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 4h volume confirmation (>2.0x 20-period average).
# Long when price breaks above Donchian upper (20-period high) AND close > 12h EMA50 (bullish trend) AND volume > 2.0x 20-period average.
# Short when price breaks below Donchian lower (20-period low) AND close < 12h EMA50 (bearish trend) AND volume > 2.0x 20-period average.
# Exit when price retests the 20-period midpoint (mean reversion) or opposite Donchian level touched.
# Uses 12h HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (2.0x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 4h timeframe.
# Donchian breakouts provide structure in both bull and bear markets when combined with HTF trend filter and volume filter.

name = "4h_Donchian20_Breakout_12hEMA50_4hVolumeConfirm_v2"
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
    # 4h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) - trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper AND close > 12h EMA50 (bullish trend) AND volume confirm
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower AND close < 12h EMA50 (bearish trend) AND volume confirm
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests Donchian midpoint (mean reversion) OR touches Donchian lower (opposite level)
            if (close[i] <= donchian_mid[i] or 
                close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests Donchian midpoint (mean reversion) OR touches Donchian upper (opposite level)
            if (close[i] >= donchian_mid[i] or 
                close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals