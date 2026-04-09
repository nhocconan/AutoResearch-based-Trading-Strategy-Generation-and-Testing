#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h and 1d Camarilla pivot levels with volume confirmation
# Trade only during 08-20 UTC session to reduce noise
# Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
# Requires volume > 1.5x 20-period average for entry
# Fixed position size 0.20 to minimize fee churn
# Target: 15-30 trades/year on 1h (60-120 total over 4 years)

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla pivot levels
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r4_4h = close_4h + range_4h * 1.1 / 2.0
    r3_4h = close_4h + range_4h * 1.1 / 4.0
    s3_4h = close_4h - range_4h * 1.1 / 4.0
    s4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align all HTF data to 1h timeframe
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r4_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Fixed position size to minimize fee churn
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit on retracement to S3 (4h or 1d) or stop at S4 breakdown (4h or 1d)
            if close[i] < s3_4h_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_4h_aligned[i] or close[i] < s4_1d_aligned[i]:  # Stop loss at S4 breakdown
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 (4h or 1d) or stop at R4 breakout (4h or 1d)
            if close[i] > r3_4h_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_4h_aligned[i] or close[i] > r4_1d_aligned[i]:  # Stop loss at R4 breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla pivot trading with volume and session confirmation
            # Fade at R3/S3 (mean reversion), breakout at R4/S4 (trend following)
            if volume_confirmed:
                # Fade at R3 (sell at resistance, expect reversion to pivot)
                if (close[i] > r3_4h_aligned[i] and close[i] < r4_4h_aligned[i]) or \
                   (close[i] > r3_1d_aligned[i] and close[i] < r4_1d_aligned[i]):
                    position = -1
                    signals[i] = -position_size
                # Fade at S3 (buy at support, expect reversion to pivot)
                elif (close[i] < s3_4h_aligned[i] and close[i] > s4_4h_aligned[i]) or \
                     (close[i] < s3_1d_aligned[i] and close[i] > s4_1d_aligned[i]):
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at R4 (buy break above resistance)
                elif close[i] > r4_4h_aligned[i] or close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout continuation at S4 (sell break below support)
                elif close[i] < s4_4h_aligned[i] or close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals