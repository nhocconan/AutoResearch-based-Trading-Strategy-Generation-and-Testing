#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot reversal with volume confirmation.
# Long when price crosses above weekly S3 level with volume > 2x average and weekly trend up.
# Short when price crosses below weekly R3 level with volume > 2x average and weekly trend down.
# Exit when price crosses weekly pivot (S1/R1) or volume drops below average.
# Uses 1w for pivot levels/trend, 1d for entry/exit. Target: 15-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week)
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    # S1 = C - (H - L) * 1.0833
    s1_1w = close_1w - range_1w * 1.0833
    # S2 = C - (H - L) * 1.1666
    s2_1w = close_1w - range_1w * 1.1666
    # S3 = C - (H - L) * 1.2500
    s3_1w = close_1w - range_1w * 1.2500
    # R1 = C + (H - L) * 1.0833
    r1_1w = close_1w + range_1w * 1.0833
    # R2 = C + (H - L) * 1.1666
    r2_1w = close_1w + range_1w * 1.1666
    # R3 = C + (H - L) * 1.2500
    r3_1w = close_1w + range_1w * 1.2500
    
    # Weekly trend: close > EMA20
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_up = close_1w > ema20_1w
    weekly_down = close_1w < ema20_1w
    
    # Align weekly levels to daily timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up.astype(float))
    weekly_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_down.astype(float))
    
    # Volume filter: volume > 2x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(weekly_up_aligned[i]) or np.isnan(weekly_down_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Reversal conditions (crossing S3/R3 with weekly trend)
        bullish_setup = (price > s3_aligned[i]) and (close[i-1] <= s3_aligned[i]) and weekly_up_aligned[i]
        bearish_setup = (price < r3_aligned[i]) and (close[i-1] >= r3_aligned[i]) and weekly_down_aligned[i]
        
        # Exit conditions (crossing S1/R1 or volume drop)
        bullish_exit = (price < s1_aligned[i]) and (close[i-1] >= s1_aligned[i])
        bearish_exit = (price > r1_aligned[i]) and (close[i-1] <= r1_aligned[i])
        
        if position == 0:
            # Long: bullish reversal setup with volume
            if bullish_setup and vol_filter:
                signals[i] = size
                position = 1
            # Short: bearish reversal setup with volume
            elif bearish_setup and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish exit or volume drops
            if bearish_exit or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish exit or volume drops
            if bullish_exit or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_S3R3_Reversal_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0