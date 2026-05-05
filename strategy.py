#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Williams Alligator (Jaw/Teeth/Lips) to define trend
# + daily Donchian(20) breakout in trend direction + volume spike confirmation
# Long: price > Donchian_High(20) AND price > Alligator Teeth (8) AND volume > 2.0 * avg_volume(20)
# Short: price < Donchian_Low(20) AND price < Alligator Teeth (8) AND volume > 2.0 * avg_volume(20)
# Exit: price crosses Alligator Teeth (8) OR volume drops below average
# Uses discrete sizing 0.25 to control drawdown
# Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# Weekly Alligator provides smooth trend filter from higher timeframe
# Daily Donchian gives precise breakout entry with structure
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "1d_Alligator_Trend_Donchian20_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Alligator (SMMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:  # Need at least one completed weekly bar for SMMA13
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly Alligator: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    # SMMA is smoothed moving average ( Wilder's smoothing, alpha=1/period )
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    alligator_jaw = smma(close_1w, 13)  # Blue line
    alligator_teeth = smma(close_1w, 8)   # Red line
    alligator_lips = smma(close_1w, 5)    # Green line
    
    # Align weekly Alligator teeth to 1d timeframe (wait for completed weekly bar)
    alligator_teeth_aligned = align_htf_to_ltf(prices, df_1w, alligator_teeth)
    
    # Get 1d data ONCE before loop for Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for Donchian20
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_high, donchian_low = donchian_channels(high_1d, low_1d, 20)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(alligator_teeth_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian High(20), above Alligator Teeth, volume confirmation, in session
            if close[i] > donchian_high_aligned[i] and close[i] > alligator_teeth_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian Low(20), below Alligator Teeth, volume confirmation, in session
            elif close[i] < donchian_low_aligned[i] and close[i] < alligator_teeth_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Alligator Teeth OR volume drops below average
            if close[i] < alligator_teeth_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Alligator Teeth OR volume drops below average
            if close[i] > alligator_teeth_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals