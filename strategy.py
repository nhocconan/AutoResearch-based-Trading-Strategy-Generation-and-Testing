#!/usr/bin/env python3
"""
12h CAMARILLA PIVOT + VOLUME + CHOPPINESS REGIME
Hypothesis: CAMARILLA pivot levels from 1-day timeframe combined with volume spikes and
choppiness regime filter provide high-probability reversal entries. Works in bull markets
via mean reversion at support/resistance and in bear markets via the same logic (pivots
hold in ranging/choppy conditions). Target: 15-30 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_chop_v1"
timeframe = "12h"
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
    
    # 1-day data for CAMARILLA pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CAMARILLA pivot levels for each 1-day bar
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    daily_range = high_1d - low_1d
    # CAMARILLA levels
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    r1 = typical_price + (daily_range * 1.1 / 12)
    r2 = typical_price + (daily_range * 1.1 / 6)
    r3 = typical_price + (daily_range * 1.1 / 4)
    r4 = typical_price + (daily_range * 1.1 / 2)
    s1 = typical_price - (daily_range * 1.1 / 12)
    s2 = typical_price - (daily_range * 1.1 / 6)
    s3 = typical_price - (daily_range * 1.1 / 4)
    s4 = typical_price - (daily_range * 1.1 / 2)
    
    # Align CAMARILLA levels to 12h timeframe (shifted by 1 for completed day only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Chopiness index (14-period) on 1-day for regime filter
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index"""
        atr_sum = np.zeros_like(close_arr)
        true_range = np.zeros_like(close_arr)
        
        for i in range(len(close_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(
                    high_arr[i] - low_arr[i],
                    abs(high_arr[i] - close_arr[i-1]),
                    abs(low_arr[i] - close_arr[i-1])
                )
            true_range[i] = tr
            
        # Calculate ATR using Wilder's smoothing (equivalent to RMA)
        atr = np.zeros_like(close_arr)
        if len(close_arr) >= period:
            # First ATR value is simple average
            atr[period-1] = np.mean(true_range[:period])
            # Subsequent values using Wilder's smoothing
            for i in range(period, len(close_arr)):
                atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Calculate Chop
        chop = np.full_like(close_arr, 50.0)  # Default neutral
        for i in range(period-1, len(close_arr)):
            if atr[i] > 0:
                highest_high = np.max(high_arr[i-period+1:i+1])
                lowest_low = np.min(low_arr[i-period+1:i+1])
                if highest_high > lowest_low:
                    log_sum = np.log10(atr[i] * period / (highest_high - lowest_low))
                    chop[i] = 100 * log_sum / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy/ranging markets (CHOP > 50)
        if chop_aligned[i] <= 50:
            # In trending markets, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S2 (strong support) or closes above R2
            if (low[i] <= s2_aligned[i] or close[i] > r2_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R2 (strong resistance) or closes below S2
            if (high[i] >= r2_aligned[i] or close[i] < s2_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long setup: price touches or penetrates S3 level with volume spike
            if (low[i] <= s3_aligned[i] and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short setup: price touches or penetrates R3 level with volume spike
            elif (high[i] >= r3_aligned[i] and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals