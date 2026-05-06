#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA trend filter
# Long when price breaks above 4h Donchian upper (20) AND 1d EMA50 > EMA50 previous (uptrend) AND volume > 1.5 * avg_volume(20) on 1h
# Short when price breaks below 4h Donchian lower (20) AND 1d EMA50 < EMA50 previous (downtrend) AND volume > 1.5 * avg_volume(20) on 1h
# Exit when price crosses 4h Donchian midpoint (mean reversion) OR opposite breakout occurs
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides structural breakout signals with proven edge
# 1d EMA50 trend filter ensures we trade with dominant daily trend (avoids counter-trend whipsaws)
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "1h_4hDonchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed 1d bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper, 1d EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > donchian_upper_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower, 1d EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h Donchian middle (mean reversion) OR opposite breakout
            if (close[i] < donchian_middle_aligned[i] or 
                close[i] < donchian_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h Donchian middle (mean reversion) OR opposite breakout
            if (close[i] > donchian_middle_aligned[i] or 
                close[i] > donchian_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals