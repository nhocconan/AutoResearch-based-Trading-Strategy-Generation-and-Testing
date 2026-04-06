#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily timeframe with weekly trend filter for directional bias.
# Uses weekly Donchian(20) breakout + 12h EMA(20) trend filter + volume confirmation (1.5x avg).
# Weekly trend filter ensures only trading with the dominant weekly trend.
# Volume filter reduces false breakouts.
# Target: 40-80 total trades over 4 years (10-20/year) to minimize fee drag.
# Daily timeframe provides sufficient signal frequency while weekly filter ensures trend alignment.

name = "1d_weekly_donchian20_12hema20_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-period) for breakout signals
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_1w = align_htf_to_ltf(prices, df_1w, high_20_1w)
    donchian_low_1w = align_htf_to_ltf(prices, df_1w, low_20_1w)
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2
    
    # 12h EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_1w[i]) or np.isnan(donchian_low_1w[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to weekly Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid_1w[i] or close[i] < donchian_low_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to weekly Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid_1w[i] or close[i] > donchian_high_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: weekly Donchian breakout + 12h EMA trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high_1w[i] and close[i] > ema_20_aligned[i]:
                    # Bullish breakout above weekly Donchian high with 12h uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low_1w[i] and close[i] < ema_20_aligned[i]:
                    # Bearish breakdown below weekly Donchian low with 12h downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals