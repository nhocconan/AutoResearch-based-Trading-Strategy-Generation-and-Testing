#!/usr/bin/env python3
# 1D_Donchian_Breakout_TrendFilter_WeeklyTrend
# Hypothesis: On daily timeframe, enter long when price breaks above 20-day Donchian high with weekly uptrend (close > weekly EMA50) and volume > 2x 20-day average.
# Enter short when price breaks below 20-day Donchian low with weekly downtrend (close < weekly EMA50) and volume > 2x average.
# Exit when price returns to opposite Donchian level or weekly trend reverses.
# Uses weekly EMA50 for trend to avoid whipsaws in both bull and bear markets.
# Targets 15-25 trades per year on 1d timeframe with position size 0.25 to minimize fee drag.

name = "1D_Donchian_Breakout_TrendFilter_WeeklyTrend"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate Donchian channels (20-period) on daily data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # Track bars since last exit to prevent churn
    
    start_idx = max(50, 20)  # Warmup for weekly EMA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_weekly_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_weekly_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_weekly_aligned[i]
        
        if position == 0:
            # Require at least 1 day since last exit to prevent churn
            if bars_since_exit >= 1:
                # Long entry: price breaks above Donchian high with weekly uptrend and volume spike
                if (close[i] > donchian_high[i] and 
                    price_above_weekly_ema and 
                    volume[i] > vol_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_exit = 0
                # Short entry: price breaks below Donchian low with weekly downtrend and volume spike
                elif (close[i] < donchian_low[i] and 
                      price_below_weekly_ema and 
                      volume[i] > vol_threshold[i]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Long exit: price returns to Donchian low or weekly trend reverses to downtrend
            if (close[i] < donchian_low[i] or 
                price_below_weekly_ema):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian high or weekly trend reverses to uptrend
            if (close[i] > donchian_high[i] or 
                price_above_weekly_ema):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
    
    return signals