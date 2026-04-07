#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily data with volume confirmation
# Fades at R3/S3 (mean reversion), breaks out at R4/S4 (trend continuation)
# Uses weekly trend filter to align with higher timeframe direction
# Designed for low frequency: target 20-40 trades/year to minimize fee drag
# Works in both bull and bear markets by adapting to regime via weekly trend

name = "6h_camarilla_pivot_1d_weekly_trend_volume_v1"
timeframe = "6h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = pd.Series(df_1w['close'].values)
    ema_1w = close_1w.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily range for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R4, R3, S3, S4
    # Based on previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + range_1d * 1.500
    r3 = prev_close + range_1d * 1.250
    s3 = prev_close - range_1d * 1.250
    s4 = prev_close - range_1d * 1.500
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime from weekly trend
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = close[i] < s3_aligned[i]
        exit_short = close[i] > r3_aligned[i]
        
        if position == 1:  # Long position
            # Exit at S3 (mean reversion) or trend reversal
            if exit_long or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit at R3 (mean reversion) or trend reversal
            if exit_short or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: bounce from S3/S4 with uptrend + volume
            if (close[i] >= s3_aligned[i] and close[i] <= s4_aligned[i]) and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: rejection at R3/R4 with downtrend + volume
            elif (close[i] <= r3_aligned[i] and close[i] >= r4_aligned[i]) and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
            # Breakout entries: close beyond R4/S4 with trend + volume
            elif close[i] > r4_aligned[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_aligned[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals