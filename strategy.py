#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with 1d trend filter (price > 1d EMA50) and volume confirmation (>2.0x 20-period average) capture institutional breakout moves in both bull and bear markets. The Camarilla levels act as magnetic pivot points where R3/S3 represent strong support/resistance and breaks indicate momentum continuation. Volume spike confirms institutional participation. Designed for ~60-100 total trades over 4 years (15-25/year) via tight entry conditions requiring multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # need for EMA20 and volume
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels
    # Camarilla levels based on previous day's range
    # R4 = close + 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1/2
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first day's values to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 2.0x 20-period average (more stringent)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d close aligned for trend filter comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 1)  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get aligned values
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        r4_val = R4_aligned[i]
        s4_val = S4_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_1d_val = close_1d_aligned[i]
        
        # 1d trend filter: price vs EMA50
        is_uptrend = close_1d_val > ema_50_val
        
        if position == 0:
            # Look for entry signals
            if is_uptrend:
                # Long conditions: price breaks above R3 with volume spike
                long_signal = (close[i] > r3_val) and vol_spike[i]
            else:
                # Short conditions: price breaks below S3 with volume spike
                short_signal = (close[i] < s3_val) and vol_spike[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price retraces back below S3 (failed breakout)
            # 2. Trend reverses (price < EMA50 on 1d)
            if close[i] < s3_val or close_1d_val < ema_50_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price retraces back above R3 (failed breakout)
            # 2. Trend reverses (price > EMA50 on 1d)
            if close[i] > r3_val or close_1d_val > ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0