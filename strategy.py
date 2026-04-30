#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Donchian breakout with volume confirmation and session filter.
# Uses 4h Donchian channels for trend direction, 1h only for precise entry timing.
# Volume confirmation (>1.8x 20-bar average) reduces false breakouts.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete position sizing at ±0.20 to minimize fee churn.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits when price retests channel mid-point.

name = "1h_Donchian4h_Breakout_VolumeConfirm_Session_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper: highest high over 20 periods
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over 20 periods
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian mid-point: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = donchian_high_aligned[i]
        curr_low = donchian_low_aligned[i]
        curr_mid = donchian_mid_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 4h Donchian high, volume spike, in session
            if (curr_close > curr_high and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low, volume spike, in session
            elif (curr_close < curr_low and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price retests Donchian mid-point (mean reversion in bear)
            if curr_close < curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price retests Donchian mid-point (mean reversion in bear)
            if curr_close > curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals