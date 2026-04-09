#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation and session filter
# 4h/1d Camarilla levels act as major support/resistance in both bull and bear markets
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend following)
# Volume confirmation (current 1h volume > 1.5x 20-period average) filters false signals
# Session filter (08-20 UTC) reduces noise trades
# Position size fixed at 0.20 to control risk and minimize fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe

name = "1h_4h_1d_camarilla_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r4_4h = close_4h + range_4h * 1.1 / 2.0
    r3_4h = close_4h + range_4h * 1.1 / 4.0
    s3_4h = close_4h - range_4h * 1.1 / 4.0
    s4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(r4_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on retracement to S3 or stop at S4 breakdown (use tighter of 4h/1d levels)
            s3_level = max(s3_4h_aligned[i], s3_1d_aligned[i])  # Higher support level
            s4_level = min(s4_4h_aligned[i], s4_1d_aligned[i])  # Lower support level
            
            if close[i] < s3_level:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_level:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 or stop at R4 breakout (use tighter of 4h/1d levels)
            r3_level = min(r3_4h_aligned[i], r3_1d_aligned[i])  # Lower resistance level
            r4_level = min(r4_4h_aligned[i], r4_1d_aligned[i])  # Lower resistance level
            
            if close[i] > r3_level:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_level:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Camarilla pivot trading with volume confirmation
            # Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
            if volume_confirmed:
                # Use 4h levels for entry timing, 1d for stronger signals
                r3_4h = r3_4h_aligned[i]
                r4_4h = r4_4h_aligned[i]
                s3_4h = s3_4h_aligned[i]
                s4_4h = s4_4h_aligned[i]
                
                r3_1d = r3_1d_aligned[i]
                r4_1d = r4_1d_aligned[i]
                s3_1d = s3_1d_aligned[i]
                s4_1d = s4_1d_aligned[i]
                
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if close[i] > r3_4h and close[i] < r4_4h:
                    position = -1
                    signals[i] = -0.20
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif close[i] < s3_4h and close[i] > s4_4h:
                    position = 1
                    signals[i] = 0.20
                # Breakout continuation at R4 (buy break above resistance)
                elif close[i] > r4_4h and close[i] > r4_1d:
                    position = 1
                    signals[i] = 0.20
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_4h and close[i] < s4_1d:
                    position = -1
                    signals[i] = -0.20
    
    return signals