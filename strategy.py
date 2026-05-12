#!/usr/bin/env python3
# 1d_WeeklyDonchianBreakout
# Hypothesis: Buy breakout above weekly Donchian high (20-week) in uptrend, sell breakout below weekly Donchian low in downtrend.
# Uses weekly price structure for trend and key levels, with daily volume confirmation.
# Designed for very low frequency (<15 trades/year) to avoid fee drag. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).

name = "1d_WeeklyDonchianBreakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Breakout above weekly Donchian high in uptrend with volume
            if (close[i] > donchian_high_20_aligned[i] and 
                trend_up and vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly Donchian low in downtrend with volume
            elif (close[i] < donchian_low_20_aligned[i] and 
                  trend_down and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below weekly Donchian low or trend reverses
            if (close[i] < donchian_low_20_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly Donchian high or trend reverses
            if (close[i] > donchian_high_20_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals