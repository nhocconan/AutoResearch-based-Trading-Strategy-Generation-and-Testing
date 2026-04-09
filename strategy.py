#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation
# 4h/1d Camarilla pivots provide multi-timeframe structure aligned with 1h timeframe
# Volume confirmation (current 1h volume > 1.8x 20-period average) filters false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# Works in bull/bear: price reacts to 4h/1d structure, volume confirms validity
# Discrete position sizing: 0.0, ±0.20 to minimize fee churn

name = "1h_4h_1d_camarilla_volume_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
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
    camarilla_r3_4h = close_4h + range_4h * 1.1 / 4.0
    camarilla_r4_4h = close_4h + range_4h * 1.1 / 2.0
    camarilla_s3_4h = close_4h - range_4h * 1.1 / 4.0
    camarilla_s4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3_1d = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(r4_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.8x average 1h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on Camarilla S3 retracement from either 4h or 1d (mean reversion)
            if close[i] < s3_4h_aligned[i] or close[i] < s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit on Camarilla R3 retracement from either 4h or 1d (mean reversion)
            if close[i] > r3_4h_aligned[i] or close[i] > r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout trading with volume confirmation and multi-timeframe alignment
            # Require both 4h and 1d levels to agree for stronger signal
            r4_agree = r4_4h_aligned[i] <= r4_1d_aligned[i]  # 4h R4 <= 1d R4
            s4_agree = s4_4h_aligned[i] >= s4_1d_aligned[i]  # 4h S4 >= 1d S4
            
            if volume_confirmed and r4_agree and s4_agree:
                # Long on Camarilla R4 breakout (bullish breakout)
                if close[i] > r4_4h_aligned[i] and close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short on Camarilla S4 breakout (bearish breakout)
                elif close[i] < s4_4h_aligned[i] and close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals