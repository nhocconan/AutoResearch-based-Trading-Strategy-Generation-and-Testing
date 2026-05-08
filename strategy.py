#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend and Donchian breakout
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Donchian channels for trend (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Daily Donchian breakout (20-period)
    donchian_high_daily = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_daily = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above daily Donchian high AND weekly trend is up (price above weekly Donchian mid) AND volume confirmation
            weekly_mid = (donchian_high_1w_aligned[i] + donchian_low_1w_aligned[i]) / 2
            long_cond = (close[i] > donchian_high_daily[i]) and (close[i] > weekly_mid) and (volume[i] > 1.5 * volume_ma[i])
            
            # Short entry: price breaks below daily Donchian low AND weekly trend is down (price below weekly Donchian mid) AND volume confirmation
            short_cond = (close[i] < donchian_low_daily[i]) and (close[i] < weekly_mid) and (volume[i] > 1.5 * volume_ma[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below daily Donchian low OR weekly trend turns down
            if (close[i] < donchian_low_daily[i]) or (close[i] < weekly_mid):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above daily Donchian high OR weekly trend turns up
            if (close[i] > donchian_high_daily[i]) or (close[i] > weekly_mid):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above daily Donchian high, price is above weekly Donchian midpoint (indicating uptrend), and volume confirms.
# Short when price breaks below daily Donchian low, price is below weekly Donchian midpoint (indicating downtrend), and volume confirms.
# Exits when price breaks opposite Donchian level or weekly trend reverses.
# Weekly trend filter prevents counter-trend trades in strong trends.
# Volume confirmation ensures breakouts have conviction.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee decay.
# Works in both bull and bear markets by following the weekly trend.