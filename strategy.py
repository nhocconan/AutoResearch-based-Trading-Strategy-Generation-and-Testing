#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Weekly pivot levels (R4/R3/S3/S4) from prior week determine bias: long above weekly R3, short below weekly S3
# Donchian(20) breakout on 6f timeframe provides entry timing in direction of weekly bias
# Volume spike (1.5x 20-period average) confirms breakout strength
# Works in bull via breakouts above weekly resistance with uptrend bias, in bear via breakdowns below weekly support with downtrend bias
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete sizing 0.25 to minimize fee churn

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate weekly pivot points from prior week (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Calculate support/resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    weekly_r4 = weekly_high + 3 * (weekly_pivot - weekly_low)
    weekly_s4 = weekly_low - 3 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar close)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # Calculate Donchian(20) channels on 6h timeframe
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_weekly_r3 = weekly_r3_aligned[i]
        curr_weekly_s3 = weekly_s3_aligned[i]
        curr_weekly_r4 = weekly_r4_aligned[i]
        curr_weekly_s4 = weekly_s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Long entry: price breaks above Donchian high AND above weekly R3 (bullish bias)
                if curr_close > curr_donchian_high and curr_close > curr_weekly_r3:
                    signals[i] = 0.25
                    position = 1
                # Short entry: price breaks below Donchian low AND below weekly S3 (bearish bias)
                elif curr_close < curr_donchian_low and curr_close < curr_weekly_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian low OR weekly S4 (strong bearish signal)
            if curr_close < curr_donchian_low or curr_close < curr_weekly_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high OR weekly R4 (strong bullish signal)
            if curr_close > curr_donchian_high or curr_close > curr_weekly_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals