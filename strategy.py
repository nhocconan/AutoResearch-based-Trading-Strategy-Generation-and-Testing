#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d weekly pivot points (Camarilla) with volume confirmation
# Weekly pivot levels act as strong support/resistance due to institutional order clustering
# Fade at R3/S3 levels (mean reversion) with volume spike confirmation
# Breakout continuation at R4/S4 levels (momentum) with volume confirmation
# 6h timeframe reduces noise while capturing multi-day moves
# Target: 12-37 trades/year (50-150 total over 4 years) with position size 0.25

name = "6h_1d_weekly_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Camarilla pivot levels (using prior week's OHLC)
    # We need to group daily data into weeks (Mon-Sun)
    # Create DataFrame for weekly resampling
    df_1d_for_weekly = pd.DataFrame({
        'high': high_1d,
        'low': low_1d,
        'close': close_1d
    }, index=pd.to_datetime(df_1d.index))
    
    # Resample to weekly (Monday start)
    weekly = df_1d_for_weekly.resample('W-MON', label='left', closed='left').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Need open price for weekly - use first day's open of the week
    # We'll approximate using prior week's close as open for simplicity
    weekly_open = weekly['close'].shift(1)
    weekly_high = weekly['high']
    weekly_low = weekly['low']
    weekly_close = weekly['close']
    
    # Calculate Camarilla levels for each week
    # Camarilla formula: 
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    weekly_range = weekly_high - weekly_low
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    r4 = weekly_close + (weekly_range * 1.1 / 2)
    r3 = weekly_close + (weekly_range * 1.1 / 4)
    r2 = weekly_close + (weekly_range * 1.1 / 6)
    r1 = weekly_close + (weekly_range * 1.1 / 12)
    s1 = weekly_close - (weekly_range * 1.1 / 12)
    s2 = weekly_close - (weekly_range * 1.1 / 6)
    s3 = weekly_close - (weekly_range * 1.1 / 4)
    s4 = weekly_close - (weekly_range * 1.1 / 2)
    
    # Create arrays for each level (same length as weekly data)
    weekly_levels = {
        'r4': r4.values,
        'r3': r3.values,
        'r2': r2.values,
        'r1': r1.values,
        'pp': weekly_pp.values,
        's1': s1.values,
        's2': s2.values,
        's3': s3.values,
        's4': s4.values
    }
    
    # Align weekly levels to 1d timeframe (forward fill from weekly close)
    aligned_1d_levels = {}
    for level_name, level_vals in weekly_levels.items():
        # Create series with weekly index, then resample to daily forward fill
        weekly_idx = weekly.index
        weekly_series = pd.Series(level_vals, index=weekly_idx)
        # Resample to daily frequency, forward fill
        daily_series = weekly_series.reindex(df_1d.index, method='ffill')
        aligned_1d_levels[level_name] = daily_series.values
    
    # Align 1d levels to 6h timeframe
    aligned_6h_levels = {}
    for level_name, level_vals in aligned_1d_levels.items():
        aligned_6h_levels[level_name] = align_htf_to_ltf(prices, df_1d, level_vals)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(aligned_6h_levels['r3'][i]) or np.isnan(aligned_6h_levels['s3'][i]) or
            np.isnan(aligned_6h_levels['r4'][i]) or np.isnan(aligned_6h_levels['s4'][i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on retracement to weekly pivot point or stop loss
            pp_level = aligned_6h_levels['pp'][i]
            if close[i] < pp_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to weekly pivot point or stop loss
            pp_level = aligned_6h_levels['pp'][i]
            if close[i] > pp_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            if volume_confirmed:
                r3 = aligned_6h_levels['r3'][i]
                s3 = aligned_6h_levels['s3'][i]
                r4 = aligned_6h_levels['r4'][i]
                s4 = aligned_6h_levels['s4'][i]
                
                # Fade at R3/S3 (mean reversion)
                if close[i] > r3 and close[i] < r4:
                    # Near R3, expect reversal down
                    position = -1
                    signals[i] = -position_size
                elif close[i] < s3 and close[i] > s4:
                    # Near S3, expect reversal up
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at R4/S4
                elif close[i] > r4:
                    # Break above R4, expect continuation up
                    position = 1
                    signals[i] = position_size
                elif close[i] < s4:
                    # Break below S4, expect continuation down
                    position = -1
                    signals[i] = -position_size
    
    return signals