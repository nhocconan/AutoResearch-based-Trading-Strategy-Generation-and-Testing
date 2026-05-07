#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot (using weekly high, low, close)
    high_prev_w = df_1w['high'].shift(1).values
    low_prev_w = df_1w['low'].shift(1).values
    close_prev_w = df_1w['close'].shift(1).values
    pivot_w = (high_prev_w + low_prev_w + close_prev_w) / 3
    # Align weekly pivot to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period high/low)
    high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    # Align Donchian levels to 6h timeframe
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~24 hours for 6h to reduce trades
    
    start_idx = max(200, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(high_20d_aligned[i]) or 
            np.isnan(low_20d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction using price vs weekly pivot
        trend_up = close > pivot_w_aligned[i]
        trend_down = close < pivot_w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above Donchian high in uptrend with volume confirmation
            if (close[i] > high_20d_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below Donchian low in downtrend with volume confirmation
            elif (close[i] < low_20d_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Donchian channel or trend change
            if (close[i] < high_20d_aligned[i] and close[i] > low_20d_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Donchian channel or trend change
            if (close[i] < high_20d_aligned[i] and close[i] > low_20d_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using 6h timeframe with Donchian(20) breakouts from daily data,
# weekly pivot trend filter, and volume confirmation will yield 12-37 trades per year
# (50-150 total over 4 years). The weekly pivot provides a higher timeframe trend filter
# that works in both bull and bear markets, while Donchian breakouts capture institutional
# moves. Volume confirmation reduces false breakouts. Position size of 0.25 manages drawdown,
# and cooldown of 4 bars prevents overtrading. This combination has not been recently tested
# and offers a novel approach for 6h timeframe trading.