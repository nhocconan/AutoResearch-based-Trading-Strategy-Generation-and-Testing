#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels with volume confirmation and session filter
# 4h Camarilla levels (R3/S3, R4/S4) act as major support/resistance that work in both bull and bear markets
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend following)
# Volume confirmation (current 1h volume > 1.5x 20-period average) filters false signals
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Position size fixed at 0.20 to minimize fee churn
# Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years)

name = "1h_4h_camarilla_volume_session_v1"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r4_4h = close_4h + range_4h * 1.1 / 2.0
    r3_4h = close_4h + range_4h * 1.1 / 4.0
    s3_4h = close_4h - range_4h * 1.1 / 4.0
    s4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Align all HTF data to 1h timeframe
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit on retracement to S3 or stop at S4 breakdown
            if close[i] < s3_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_4h_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 or stop at R4 breakout
            if close[i] > r3_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_4h_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla pivot trading with volume and session confirmation
            # Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
            if volume_confirmed:
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if close[i] > r3_4h_aligned[i] and close[i] < r4_4h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif close[i] < s3_4h_aligned[i] and close[i] > s4_4h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at R4 (buy break above resistance)
                elif close[i] > r4_4h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_4h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals