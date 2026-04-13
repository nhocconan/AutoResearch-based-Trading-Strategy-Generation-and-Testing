#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels with 1d trend filter and volume confirmation.
# Camarilla pivots provide precise support/resistance levels based on prior day's range.
# In strong trends, price tends to respect S3/R3 levels; in ranges, it oscillates between S1/R1.
# Combined with 1d trend filter (EMA50) and volume spikes, it filters false breakouts.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_r2 = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_s2 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            H = high_1d[i-1]
            L = low_1d[i-1]
            C = close_1d[i-1]
            range_hl = H - L
            
            camarilla_r4[i] = C + (range_hl * 1.1 / 2)
            camarilla_r3[i] = C + (range_hl * 1.1 / 4)
            camarilla_r2[i] = C + (range_hl * 1.1 / 6)
            camarilla_r1[i] = C + (range_hl * 1.1 / 12)
            camarilla_s1[i] = C - (range_hl * 1.1 / 12)
            camarilla_s2[i] = C - (range_hl * 1.1 / 6)
            camarilla_s3[i] = C - (range_hl * 1.1 / 4)
            camarilla_s4[i] = C - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Average volume (4-period = 2 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(4, n):
        avg_volume[i] = np.mean(volume[i-4:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(4, n):
        # Skip if any required data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price crosses above R3 with volume + above 1d EMA50
            if (price > r3 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below S3 with volume + below 1d EMA50
            elif (price < s3 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below R1 or trend changes
            if (price < r1 or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above S1 or trend changes
            if (price > s1 or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_CamarillaPivot_Trend_Volume"
timeframe = "12h"
leverage = 1.0