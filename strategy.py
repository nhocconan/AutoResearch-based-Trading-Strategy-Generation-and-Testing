#!/usr/bin/env python3
# 12h_Donchian_20_1dTrend_VolumeFilter
# Hypothesis: 12h Donchian channel breakout with 1d EMA trend filter and volume confirmation.
# Breakouts above upper Donchian (20) signal bullish momentum; breakdowns below lower Donchian (20) signal bearish momentum.
# Only trade breakouts aligned with daily trend (price > EMA34 for longs, price < EMA34 for shorts).
# Volume must be above 1.5x 20-period average to confirm breakout strength.
# Exits when price returns to the Donchian midpoint or trend reverses.
# Targets 15-30 trades/year to minimize fee drag while capturing significant moves.

name = "12h_Donchian_20_1dTrend_VolumeFilter"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # Warmup for Donchian, daily EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + uptrend + volume spike
            if close[i] > donchian_high[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + downtrend + volume spike
            elif close[i] < donchian_low[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or trend reversal
            if close[i] < donchian_mid[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or trend reversal
            if close[i] > donchian_mid[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals