#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Trend_Filter
Hypothesis: Weekly Donchian breakout with 1-day trend filter.
In bull markets (price > daily EMA50), long on weekly high breakout.
In bear markets (price < daily EMA50), short on weekly low breakout.
Target: 10-25 trades per year (~40-100 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels: 20-week high/low
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate 20-period rolling high/low on weekly data
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from daily EMA50
        uptrend_regime = close[i] > ema_50[i]
        downtrend_regime = close[i] < ema_50[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high in uptrend regime
            long_entry = (close[i] > donchian_high_aligned[i]) and uptrend_regime
            # Short: price breaks below weekly Donchian low in downtrend regime
            short_entry = (close[i] < donchian_low_aligned[i]) and downtrend_regime
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below daily EMA50 (trend change)
            if not uptrend_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above daily EMA50 (trend change)
            if not downtrend_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals