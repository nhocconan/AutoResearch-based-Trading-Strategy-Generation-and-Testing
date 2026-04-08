#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Donchian(20) breakouts aligned with weekly pivot trend and volume spikes
capture strong momentum moves while filtering false breakouts. Works in bull/bear
by requiring weekly trend alignment. Targets 15-30 trades/year.
"""
name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter - call ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detector: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly EMA for current 6h bar
        ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)[i]
        
        # Trend filter: weekly EMA50 direction
        price_above_weekly_ema = close[i] > ema50_1w_aligned
        price_below_weekly_ema = close[i] < ema50_1w_aligned
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR trend turns bearish
            if close[i] <= lowest_low[i] or not price_above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR trend turns bullish
            if close[i] >= highest_high[i] or not price_below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike and trend alignment
            if volume_spike[i] and price_above_weekly_ema and close[i] >= highest_high[i]:
                position = 1
                signals[i] = 0.25
            elif volume_spike[i] and price_below_weekly_ema and close[i] <= lowest_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals