#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance. 
Breakout above R3 or below S3 with volume confirmation and daily trend filter captures 
institutional breakout moves. Designed for low trade frequency (20-40/year) to minimize 
fee drag and work in both bull and bear markets by following the daily trend.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the day.
    Based on previous day's high, low, close.
    R4 = close + (high - low) * 1.5000
    R3 = close + (high - low) * 1.2500
    R2 = close + (high - low) * 1.1666
    R1 = close + (high - low) * 1.0833
    PP = (high + low + close) / 3
    S1 = close - (high - low) * 1.0833
    S2 = close - (high - low) * 1.1666
    S3 = close - (high - low) * 1.2500
    S4 = close - (high - low) * 1.5000
    """
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r4 = close + range_hl * 1.5000
    r3 = close + range_hl * 1.2500
    r2 = close + range_hl * 1.1666
    r1 = close + range_hl * 1.0833
    pp = pivot
    s1 = close - range_hl * 1.0833
    s2 = close - range_hl * 1.1666
    s3 = close - range_hl * 1.2500
    s4 = close - range_hl * 1.5000
    
    return r1, r2, r3, r4, s1, s2, s3, s4, pp

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from daily data (using previous day's data)
    r1 = np.full_like(close, np.nan)
    r2 = np.full_like(close, np.nan)
    r3 = np.full_like(close, np.nan)
    r4 = np.full_like(close, np.nan)
    s1 = np.full_like(close, np.nan)
    s2 = np.full_like(close, np.nan)
    s3 = np.full_like(close, np.nan)
    s4 = np.full_like(close, np.nan)
    pp = np.full_like(close, np.nan)
    
    # We need to shift the daily data by 1 to use previous day's levels
    # For each 4h bar, we use the Camarilla levels from the previous completed day
    if len(df_1d) >= 2:
        # Calculate Camarilla for each day using previous day's data
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Only calculate where we have valid previous day data
        valid_idx = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
        if np.any(valid_idx):
            # Calculate Camarilla levels for valid indices
            for i in range(len(df_1d)):
                if valid_idx[i]:
                    ph, pl, pc = prev_high[i], prev_low[i], prev_close[i]
                    if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
                        r1_4h, r2_4h, r3_4h, r4_4h, s1_4h, s2_4h, s3_4h, s4_4h, pp_4h = calculate_camarilla(ph, pl, pc)
                        # These levels are valid for the entire next day
                        # We'll assign them to all 4h bars that fall within this day
                        pass  # We'll handle this in the alignment step
    
    # Calculate Camarilla levels using the helper function on arrays
    # We need to get the values for each day and then align
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Calculate Camarilla levels where we have valid data
        r1_vals = np.full_like(prev_close, np.nan)
        r2_vals = np.full_like(prev_close, np.nan)
        r3_vals = np.full_like(prev_close, np.nan)
        r4_vals = np.full_like(prev_close, np.nan)
        s1_vals = np.full_like(prev_close, np.nan)
        s2_vals = np.full_like(prev_close, np.nan)
        s3_vals = np.full_like(prev_close, np.nan)
        s4_vals = np.full_like(prev_close, np.nan)
        pp_vals = np.full_like(prev_close, np.nan)
        
        valid_mask = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
        if np.any(valid_mask):
            for i in range(len(prev_close)):
                if valid_mask[i]:
                    ph, pl, pc = prev_high[i], prev_low[i], prev_close[i]
                    r1_val, r2_val, r3_val, r4_val, s1_val, s2_val, s3_val, s4_val, pp_val = calculate_camarilla(ph, pl, pc)
                    r1_vals[i] = r1_val
                    r2_vals[i] = r2_val
                    r3_vals[i] = r3_val
                    r4_vals[i] = r4_val
                    s1_vals[i] = s1_val
                    s2_vals[i] = s2_val
                    s3_vals[i] = s3_val
                    s4_vals[i] = s4_val
                    pp_vals[i] = pp_val
    
    # Align the daily Camarilla levels to 4h timeframe
    # Each day's levels apply to all 4h bars of that day
    if len(df_1d) >= 2 and np.any(valid_mask):
        r1_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], r1_vals[1:])  # Skip first as it's NaN due to shift
        r2_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], r2_vals[1:])
        r3_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], r3_vals[1:])
        r4_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], r4_vals[1:])
        s1_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], s1_vals[1:])
        s2_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], s2_vals[1:])
        s3_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], s3_vals[1:])
        s4_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], s4_vals[1:])
        pp_4h = align_htf_to_ltf(prices, df_1d.iloc[1:], pp_vals[1:])
    else:
        # Return zeros if we can't calculate
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection: current volume > 2x average volume of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 with volume spike and daily uptrend
            if (close[i] > r3_4h[i] and 
                volume_spike[i] and 
                close[i] > ema50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and daily downtrend
            elif (close[i] < s3_4h[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns down
            if (close[i] < r3_4h[i] or close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns up
            if (close[i] > s3_4h[i] or close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals