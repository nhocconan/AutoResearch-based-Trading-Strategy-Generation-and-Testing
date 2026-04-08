#!/usr/bin/env python3

# 12h_camarilla_pivot_daily_trend_volume_v1
# Hypothesis: Camarilla pivot levels from daily timeframe combined with daily trend filter and volume confirmation.
# Enters long when price touches S3 level in daily uptrend with volume spike, short when touches R3 level in daily downtrend.
# Exits when price moves to opposite S4/R4 level or trend reverses.
# Designed for 12h timeframe to capture multi-day swings with low frequency (target: 15-25 trades/year).
# Works in both bull and bear markets by following daily trend and fading extreme intraday moves to pivot levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v1"
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
    
    # Daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous daily bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Use previous day's OHLC to avoid look-ahead
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate pivot levels
    diff = prev_high_1d - prev_low_1d
    r4 = prev_close_1d + diff * 1.1 / 2
    r3 = prev_close_1d + diff * 1.1 / 4
    r2 = prev_close_1d + diff * 1.1 / 6
    r1 = prev_close_1d + diff * 1.1 / 12
    s1 = prev_close_1d - diff * 1.1 / 12
    s2 = prev_close_1d - diff * 1.1 / 6
    s3 = prev_close_1d - diff * 1.1 / 4
    s4 = prev_close_1d - diff * 1.1 / 2
    
    # Align daily pivot levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: price reaches S4 level or trend reverses
            if close[i] <= s4_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches R4 level or trend reverses
            if close[i] >= r4_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: price touches S3 level in uptrend
                if daily_uptrend and low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches R3 level in downtrend
                elif daily_downtrend and high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals