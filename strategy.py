#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level (S3/R3) breakout with 1d volume surge filter and ADX trend filter.
# Long when price breaks above R3 with 1d ADX > 25 (trending) and 1d volume > 2x 20-day average.
# Short when price breaks below S3 with 1d ADX > 25 and 1d volume > 2x 20-day average.
# Exit when price crosses back below R3 (for long) or above S3 (for short).
# Uses proven Camarilla pivot structure with volume and trend confirmation for high-probability breakouts.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drag.

name = "4h_Camarilla_R3S3_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h Camarilla levels (based on previous day's OHLC)
    # Calculate using prior day's data to avoid look-ahead
    prev_day_close = pd.Series(close).shift(1)
    prev_day_high = pd.Series(high).shift(1)
    prev_day_low = pd.Series(low).shift(1)
    
    # Camarilla formulas: R3 = close + 1.1*(high-low)*1.1/2, S3 = close - 1.1*(high-low)*1.1/2
    # Simplified: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low)
    camarilla_s3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low)
    
    # Forward fill to handle first bar
    camarilla_r3 = pd.Series(camarilla_r3).ffill().bfill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().bfill().values
    
    # 1d data for ADX and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ADX (14-period) on 1d data
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d volume filter: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume_1d > (2.0 * vol_ma20)
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_surge_aligned = align_htf_to_ltf(prices, df_1d, volume_surge)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_surge_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, ADX > 25, volume surge
            long_cond = (close[i] > camarilla_r3[i]) and (adx_aligned[i] > 25) and volume_surge_aligned[i]
            # Short conditions: break below S3, ADX > 25, volume surge
            short_cond = (close[i] < camarilla_s3[i]) and (adx_aligned[i] > 25) and volume_surge_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross back below R3
            if close[i] < camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross back above S3
            if close[i] > camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals