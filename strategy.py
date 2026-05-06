#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels from 1w timeframe for structural reference
# Long when price breaks above weekly R3 level with 1d EMA20 > EMA50 and volume confirmation
# Short when price breaks below weekly S3 level with 1d EMA20 < EMA50 and volume confirmation
# Exit when price touches weekly pivot point (PP) or opposite S1/R1 level
# Uses weekly Camarilla for multi-week structure, 1d EMA for intermediate trend, volume for confirmation
# Designed to work in bull (breakout continuation) and bear (breakdown continuation) markets
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1wCamarilla_R3S3_Breakout_1dEMATrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # Camarilla: PP = (H + L + C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12.0
    s1_1w = close_1w - range_1w * 1.1 / 12.0
    r2_1w = close_1w + range_1w * 1.1 / 6.0
    s2_1w = close_1w - range_1w * 1.1 / 6.0
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 and EMA50
    close_series_1d = pd.Series(close_1d)
    ema_20_1d = close_series_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA values to 6h timeframe (wait for completed 1d bar)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 with 1d EMA20 > EMA50 and volume confirmation
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                ema_20_aligned[i] > ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 with 1d EMA20 < EMA50 and volume confirmation
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  ema_20_aligned[i] < ema_50_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches weekly PP or S1 (profit take or reversal)
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches weekly PP or R1 (profit take or reversal)
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals