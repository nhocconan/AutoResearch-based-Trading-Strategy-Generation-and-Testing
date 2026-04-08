#!/usr/bin/env python3
# 12h_camarilla_pivot_weekly_trend_volume_v1
# Hypothesis: Camarilla pivot levels from weekly timeframe with daily trend filter and volume confirmation on 12h chart.
# Works in bull/bear markets by using weekly pivot points as dynamic support/resistance and daily trend filter to avoid counter-trend trades.
# Target: 15-35 trades/year for low fee drag (12h timeframe).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels (Camarilla) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for weekly timeframe
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    hl_range = high_1w - low_1w
    camarilla_r4 = close_1w + hl_range * 1.1 / 2
    camarilla_r3 = close_1w + hl_range * 1.1 / 4
    camarilla_r2 = close_1w + hl_range * 1.1 / 6
    camarilla_r1 = close_1w + hl_range * 1.1 / 12
    camarilla_s1 = close_1w - hl_range * 1.1 / 12
    camarilla_s2 = close_1w - hl_range * 1.1 / 6
    camarilla_s3 = close_1w - hl_range * 1.1 / 4
    camarilla_s4 = close_1w - hl_range * 1.1 / 2
    
    # Align weekly Camarilla levels to 12h timeframe (wait for weekly bar to close)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Daily trend filter (1d EMA50) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: price touches S3 or S4 level, or trend reversal
            if close[i] <= s3_1w_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price touches R3 or R4 level, or trend reversal
            if close[i] >= r3_1w_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: price touches S1 level in uptrend with rejection (close above S1)
                if daily_uptrend and low[i] <= s1_1w_aligned[i] and close[i] > s1_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches R1 level in downtrend with rejection (close below R1)
                elif daily_downtrend and high[i] >= r1_1w_aligned[i] and close[i] < r1_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals