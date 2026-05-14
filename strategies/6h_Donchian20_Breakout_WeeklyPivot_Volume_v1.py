#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Donchian breakouts capture momentum bursts. Weekly pivot (from 1d data) provides
# institutional bias: price above weekly PP = bullish bias, below = bearish bias.
# Volume confirmation ensures breakouts have conviction. Works in bull/bear markets
# by requiring alignment with weekly pivot direction. Discrete sizing (0.25) limits
# drawdown and reduces fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "6h_Donchian20_Breakout_WeeklyPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (requires 5 days of OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's OHLC (Monday's open to Friday's close)
    # Using prior completed week: need at least 5 days
    prior_week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values
    prior_week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values
    prior_week_close = df_1d['close'].shift(1).values  # Friday's close
    
    # Weekly pivot point: PP = (High + Low + Close) / 3
    weekly_pp = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    
    # Align weekly pivot to 6h (changes only when weekly bar closes)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    
    # Donchian channels (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pp_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Donchian high, above weekly PP, volume confirm
            if price > donchian_high[i] and price > weekly_pp_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < Donchian low, below weekly PP, volume confirm
            elif price < donchian_low[i] and price < weekly_pp_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to weekly PP or below Donchian low
            if price < weekly_pp_aligned[i] or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to weekly PP or above Donchian high
            if price > weekly_pp_aligned[i] or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals