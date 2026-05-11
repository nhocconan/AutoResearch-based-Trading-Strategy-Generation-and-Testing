#!/usr/bin/env python3
# 1d_WeeklyTrend_Follow
# Hypothesis: Use 1-week trend (EMA50) to determine direction, enter on 1d breakout of 20-day Donchian channel with volume confirmation.
# In bull markets: long when price breaks above Donchian high(20) with rising weekly EMA50 and volume spike.
# In bear markets: short when price breaks below Donchian low(20) with falling weekly EMA50 and volume spike.
# Exit when price returns to Donchian midpoint or weekly trend reverses.
# Weekly trend filter prevents counter-trend trades, Donchian breakout captures momentum, volume confirms strength.

name = "1d_WeeklyTrend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
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
    
    # --- Weekly EMA50 trend ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        if i == 50:
            ema_1w[i] = np.mean(close_1w[0:50])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (50 + 1)) + (ema_1w[i-1] * (49 / (50 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_1w), np.nan)
    for i in range(51, len(close_1w)):
        ema_slope[i] = ema_1w[i] - ema_1w[i-1]
    
    # Align weekly EMA and slope to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope)
    
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
            np.isnan(ema_1w_aligned[i]) or
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
                # Long: upward breakout + rising weekly EMA50 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: downward breakout + falling weekly EMA50 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR weekly EMA50 slope turns negative
                if close[i] < donchian_mid[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR weekly EMA50 slope turns positive
                if close[i] > donchian_mid[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals