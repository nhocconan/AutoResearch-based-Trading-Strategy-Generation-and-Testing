#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume spike and choppiness regime filter.
# Enter long when price breaks above Camarilla R3 level with volume > 2x 20-bar average and CHOP > 61.8 (trending).
# Enter short when price breaks below Camarilla S3 level with volume > 2x 20-bar average and CHOP > 61.8.
# Exit when price returns to Camarilla Pivot level or opposite breakout occurs.
# Camarilla levels provide precise support/resistance from 1d timeframe.
# Volume spike confirms institutional participation.
# Choppiness filter ensures we only trade in trending markets (avoids chop).
# Discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    pivot = pd.Series(typical_price).rolling(window=1, min_periods=1).mean().values  # Use previous day's typical price
    # Shift to use previous day's data for today's levels
    pivot = np.roll(pivot, 1)
    pivot[0] = np.nan
    
    # Camarilla levels based on previous day's range
    range_1d = high_1d - low_1d
    r1 = pivot + range_1d * 1.1 / 12
    r2 = pivot + range_1d * 1.1 / 6
    r3 = pivot + range_1d * 1.1 / 4
    r4 = pivot + range_1d * 1.1 / 2
    s1 = pivot - range_1d * 1.1 / 12
    s2 = pivot - range_1d * 1.1 / 6
    s3 = pivot - range_1d * 1.1 / 4
    s4 = pivot - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 4h choppiness index: CHOP > 61.8 = trending (use 14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        # True Range for first bar
        tr[0] = high_arr[0] - low_arr[0]
        # ATR calculation
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        for i in range(period, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Highest high and lowest low over period
        highest_high = np.zeros(len(close_arr))
        lowest_low = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if i < period:
                highest_high[i] = np.max(high_arr[:i+1])
                lowest_low[i] = np.min(low_arr[:i+1])
            else:
                highest_high[i] = np.max(high_arr[i-period+1:i+1])
                lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # Chop calculation
        chop = np.full(len(close_arr), np.nan)
        for i in range(period-1, len(close_arr)):
            if np.sum(atr[i-period+1:i+1]) > 0:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / 
                                         (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop_values = calculate_chop(high, low, close, 14)
    chop_filter = chop_values > 61.8  # Trending market
    
    # Calculate 4h volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop_values[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < s3_aligned[i]
        
        # Exit conditions: return to pivot level
        long_exit = close[i] <= pivot_aligned[i]
        short_exit = close[i] >= pivot_aligned[i]
        
        # Entry conditions with volume and chop confirmation
        long_entry = long_breakout and volume_confirm[i] and chop_filter[i]
        short_entry = short_breakout and volume_confirm[i] and chop_filter[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and long_exit:
            signals[i] = 0.0
            position = 0
        elif position == -1 and short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals