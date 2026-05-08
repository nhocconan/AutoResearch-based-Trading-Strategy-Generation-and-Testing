#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + volume spike + choppiness regime
# Camarilla levels (S1/S2/S3, R1/R2/R3) derived from prior day's range act as strong support/resistance.
# Price touching S1/R1 with rejection (close back inside range) offers high-probability mean reversion.
# Volume spike confirms institutional interest at the level.
# Choppiness filter (CHOP > 61.8) ensures we only mean-revert in ranging markets, avoid trending days.
# Target: 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "4h_Camarilla_R1S1_Rejection_Volume_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R4 = C + ((H-L) * 1.500), R3 = C + ((H-L) * 1.250), R2 = C + ((H-L) * 1.166)
    # R1 = C + ((H-L) * 1.083), S1 = C - ((H-L) * 1.083)
    # S2 = C - ((H-L) * 1.166), S3 = C - ((H-L) * 1.250), S4 = C - ((H-L) * 1.500)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Prior day's values (shift by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will use invalid data, but will be filtered out by alignment
    
    # Calculate Camarilla levels
    R1 = prev_close + ((prev_high - prev_low) * 1.083)
    S1 = prev_close - ((prev_high - prev_low) * 1.083)
    
    # Align Camarilla levels to 4h timeframe (wait for prior day to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Choppiness index: determines if market is ranging (CHOP > 61.8) or trending (CHOP < 38.2)
    # Using 14-period chop
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = []
        tr = []
        for i in range(len(close_arr)):
            if i == 0:
                tr.append(high_arr[i] - low_arr[i])
            else:
                tr.append(max(high_arr[i] - low_arr[i], 
                           abs(high_arr[i] - close_arr[i-1]),
                           abs(low_arr[i] - close_arr[i-1])))
            if i < period:
                atr.append(np.nan)
            else:
                atr.append(np.mean(tr[i-period+1:i+1]))
        atr = np.array(atr)
        # Avoid division by zero
        atr_sum = np.nancumsum(atr)
        atr_sum_shift = np.roll(atr_sum, period)
        atr_sum_shift[:period] = 0
        atr_period_sum = atr_sum - atr_sum_shift
        chop = 100 * np.log10(atr_period_sum / (np.max(high_arr) - np.min(low_arr))) / np.log10(period)
        # Handle edge cases
        chop = np.where((np.max(high_arr) - np.min(low_arr)) == 0, 50, chop)
        return chop
    
    # Calculate chop on 1d data then align
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf_val = vol_conf[i]
        curr_close = close[i]
        
        if position == 0:
            # Enter long: price touches S1 and closes back above it (rejection) in choppy market with volume
            if low[i] <= s1 and curr_close > s1 and chop_val > 61.8 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 and closes back below it (rejection) in choppy market with volume
            elif high[i] >= r1 and curr_close < r1 and chop_val > 61.8 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 (breakdown) or chop drops (trending begins)
            if close[i] < s1 or chop_val < 50:  # exit if trend emerging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R1 (breakout) or chop drops (trending begins)
            if close[i] > r1 or chop_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals