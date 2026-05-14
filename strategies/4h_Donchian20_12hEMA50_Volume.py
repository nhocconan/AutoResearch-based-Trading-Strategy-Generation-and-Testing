#!/usr/bin/env python3
# 4h_Donchian20_12hEMA50_Volume
# Hypothesis: 4h Donchian(20) breakout for momentum, filtered by 12h EMA50 trend and volume spike.
# Long when: price breaks above Donchian high(20), 12h EMA50 rising, volume > 1.5x 20-period avg.
# Short when: price breaks below Donchian low(20), 12h EMA50 falling, volume > 1.5x 20-period avg.
# Exit when price crosses back to Donchian midpoint or 12h EMA50 trend reverses.
# Works in bull markets by catching breakouts and in bear by catching breakdowns with trend filter.
# Donchian provides clear structure, EMA50 filters counter-trend moves, volume confirms strength.

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian(20) channels ---
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # --- 12h EMA50 trend ---
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    for i in range(50, len(close_12h)):
        if i == 50:
            ema_12h[i] = np.mean(close_12h[0:50])
        else:
            ema_12h[i] = (close_12h[i] * 2 / (50 + 1)) + (ema_12h[i-1] * (49 / (50 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_12h), np.nan)
    for i in range(51, len(close_12h)):
        ema_slope[i] = ema_12h[i] - ema_12h[i-1]
    
    # Align 12h EMA and slope to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20), EMA50, and volume MA(20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if breakout_up and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: upward breakout + rising EMA50 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: downward breakout + falling EMA50 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR EMA50 slope turns negative
                if close[i] < donchian_mid[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR EMA50 slope turns positive
                if close[i] > donchian_mid[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals