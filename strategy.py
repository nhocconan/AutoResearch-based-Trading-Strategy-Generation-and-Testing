#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1w Donchian upper channel AND 1w EMA50 > EMA50 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below 1w Donchian lower channel AND 1w EMA50 < EMA50 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price crosses back through 1w Donchian midpoint (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1w Donchian breakouts capture strong weekly momentum while EMA50 filter ensures trend alignment
# Volume confirmation (2.0x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "12h_1wDonchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian and EMA calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) channels
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (highest_high_20 + lowest_low_20) / 2.0
    
    # Align 1w Donchian channels to 12h timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper, 1w EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > donchian_high_aligned[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower, 1w EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1w Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1w Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals