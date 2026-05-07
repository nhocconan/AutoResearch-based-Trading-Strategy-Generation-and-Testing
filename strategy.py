# The strategy aims to capture breakouts from 1-week Donchian channels, confirmed by weekly trend and volume spikes.
# It trades on the 6h timeframe to balance signal frequency and avoid excessive trading costs.
# The weekly trend filter ensures trades align with the higher timeframe momentum, improving win rate in both bull and bear markets.
# Volume confirmation adds conviction, filtering out false breakouts.
# A cooldown period prevents overtrading, keeping trade frequency within sustainable limits.

#!/usr/bin/env python3

name = "6h_WeeklyDonchianBreakout_TrendFilter"
timeframe = "6h"
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
    
    # Get weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average (on 6h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # Prevent overtrading (approx 3 days for 6h)
    
    start_idx = max(20, 50)  # Warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        trend_1w_up = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_1w_down = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above weekly Donchian high in weekly uptrend with volume confirmation
            if (close[i] > high_20_aligned[i] and 
                trend_1w_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below weekly Donchian low in weekly downtrend with volume confirmation
            elif (close[i] < low_20_aligned[i] and 
                  trend_1w_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below weekly Donchian high OR trend change
            if (close[i] < high_20_aligned[i]) or not trend_1w_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above weekly Donchian low OR trend change
            if (close[i] > low_20_aligned[i]) or not trend_1w_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakouts with weekly trend filter and volume confirmation.
# Long when price breaks above 20-week high in weekly uptrend with volume spike.
# Short when price breaks below 20-week low in weekly downtrend with volume spike.
# Weekly EMA50 filter ensures we trade with the higher timeframe trend.
# Volume spike confirms institutional participation. Cooldown reduces overtrading.
# Using 6h timeframe targets 12-30 trades/year to avoid fee drag. Works in both bull and bear markets by capturing breakouts in the direction of the weekly trend.