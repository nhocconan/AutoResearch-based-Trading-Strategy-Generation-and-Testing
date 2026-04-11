#!/usr/bin/env python3
# 6h_1d_1w_camarilla_pivot_volume_v1
# Strategy: 6h Candles breakout at daily/weekly Camarilla levels with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (from daily and weekly) act as strong support/resistance.
# Breakouts above weekly R4 or below weekly S4 with volume confirmation capture high-probability moves.
# Weekly context filters out noise; daily provides precise entry levels. Works in bull markets via long
# breakouts and bear markets via short breakdowns. Low trade frequency (~20-40/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (for context)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_range = weekly_high - weekly_low
    W_R4 = weekly_close + 1.5 * weekly_range  # Weekly resistance level 4
    W_S4 = weekly_close - 1.5 * weekly_range  # Weekly support level 4
    
    # Calculate daily Camarilla levels (for entry)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_range = daily_high - daily_low
    D_R4 = daily_close + 1.5 * daily_range  # Daily resistance level 4
    D_S4 = daily_close - 1.5 * daily_range  # Daily support level 4
    
    # Align weekly Camarilla levels to 6h timeframe
    W_R4_6h = align_htf_to_ltf(prices, df_1w, W_R4)
    W_S4_6h = align_htf_to_ltf(prices, df_1w, W_S4)
    
    # Align daily Camarilla levels to 6h timeframe
    D_R4_6h = align_htf_to_ltf(prices, df_1d, D_R4)
    D_S4_6h = align_htf_to_ltf(prices, df_1d, D_S4)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(W_R4_6h[i]) or np.isnan(W_S4_6h[i]) or np.isnan(D_R4_6h[i]) or np.isnan(D_S4_6h[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using daily levels
        breakout_up = high[i] > D_R4_6h[i-1]
        breakdown_down = low[i] < D_S4_6h[i-1]
        
        # Weekly context filter: price above weekly R4 = bullish context, below weekly S4 = bearish
        weekly_bullish = close[i] > W_R4_6h[i]
        weekly_bearish = close[i] < W_S4_6h[i]
        
        # Entry conditions
        # Long: Breakout above D_R4 AND weekly bullish context AND volume confirmation
        if breakout_up and weekly_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below D_S4 AND weekly bearish context AND volume confirmation
        elif breakdown_down and weekly_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using daily levels
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals