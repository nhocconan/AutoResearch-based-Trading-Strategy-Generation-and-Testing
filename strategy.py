#!/usr/bin/env python3
# 12h_WeeklyDonchian_Breakout_With_Volume_Confirmation
# Hypothesis: Weekly Donchian breakouts on 12h chart provide strong directional moves.
# Long when price breaks above 20-week high with volume confirmation; short when breaks below 20-week low.
# Weekly trend filter (1d EMA34) ensures alignment with higher timeframe trend.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag in ranging/ bear markets.

name = "12h_WeeklyDonchian_Breakout_With_Volume_Confirmation"
timeframe = "12h"
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # 20-period rolling max/min for Donchian channels
    weekly_high_max = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_low_min = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe (waits for weekly close)
    weekly_high_max_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_max)
    weekly_low_min_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_min)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 12h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly Donchian (20), daily EMA (34), volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_high_max_aligned[i]) or np.isnan(weekly_low_min_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter (using daily EMA as proxy for weekly trend)
        weekly_uptrend = close[i] > ema_34_1d_aligned[i]
        weekly_downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to weekly Donchian channels
        price_above_weekly_high = close[i] > weekly_high_max_aligned[i]
        price_below_weekly_low = close[i] < weekly_low_min_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + weekly uptrend + volume spike
            if price_above_weekly_high and weekly_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low + weekly downtrend + volume spike
            elif price_below_weekly_low and weekly_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly Donchian low or weekly trend turns down
            if close[i] < weekly_low_min_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly Donchian high or weekly trend turns up
            if close[i] > weekly_high_max_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals