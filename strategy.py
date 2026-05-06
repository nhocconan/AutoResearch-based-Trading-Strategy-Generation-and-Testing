#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above weekly Donchian upper (20-period high) AND 1w EMA200 > EMA200 previous 5 bars (strong uptrend) AND volume > 2.0 * avg_volume(50) on 1d
# Short when price breaks below weekly Donchian lower (20-period low) AND 1w EMA200 < EMA200 previous 5 bars (strong downtrend) AND volume > 2.0 * avg_volume(50) on 1d
# Exit when price crosses back through the weekly Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Donchian provides strong breakout levels that reduce whipsaw in ranging markets
# 1w EMA200 trend filter ensures we trade with the dominant weekly trend (avoids counter-trend)
# Volume confirmation (2.0x) validates breakout strength while limiting overtrading
# Works in both bull and bear markets by following the weekly trend

name = "1d_WeeklyDonchian20_1wEMA200_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Donchian and EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Upper = rolling max of high, Lower = rolling min of low
    high_roll_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian to daily timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 50-period average volume on 1d
    avg_volume_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * avg_volume_50)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume_50[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper, 1w EMA200 trending up (current > 5 bars ago), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                ema_200_1w_aligned[i] > ema_200_1w_aligned[i-5] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower, 1w EMA200 trending down (current < 5 bars ago), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_200_1w_aligned[i] < ema_200_1w_aligned[i-5] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below weekly Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above weekly Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals