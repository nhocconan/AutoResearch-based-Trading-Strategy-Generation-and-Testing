#!/usr/bin/env python3
# 1h_4hDonchianBreakout_1dTrend_Volume
# Hypothesis: Trade 1h breakouts of 4h Donchian channels aligned with daily trend and volume.
# Uses 4h Donchian (20) for structural breakouts, 1d EMA50 for trend filter, and volume confirmation.
# Designed for 1h timeframe with 4h directional filter to limit trades to 15-30/year.
# Works in bull markets via trend-following breakouts and in bear via short breakdowns.
# Volume filter ensures momentum confirmation, reducing false breakouts.

name = "1h_4hDonchianBreakout_1dTrend_Volume"
timeframe = "1h"
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
    
    # === 4h Donchian Channel (20) for breakout signals ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian upper/lower (20-period high/low)
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (already waits for 4h bar close)
    donchian_up = align_htf_to_ltf(prices, df_4h, high_roll)
    donchian_dn = align_htf_to_ltf(prices, df_4h, low_roll)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation (24-period average on 1h) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_dn[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_up[i]
        breakout_down = close[i] < donchian_dn[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: breakout above Donchian upper, uptrend, volume confirmation
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.20
                position = 1
            # SHORT: breakout below Donchian lower, downtrend, volume confirmation
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below Donchian lower or trend reversal
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: breakout above Donchian upper or trend reversal
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals