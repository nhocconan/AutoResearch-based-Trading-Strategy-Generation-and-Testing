#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above R3 or S3 with volume > 1.5x average and weekly close > weekly open (bullish week).
# Short when price breaks below S3 or R3 with volume > 1.5x average and weekly close < weekly open (bearish week).
# Exit when price returns to the central pivot point (PP) or reverses at opposite S/R level.
# Designed for ~15-25 trades/year per symbol (~60-100 total over 4 years).
name = "12h_Camarilla_Pivot_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    PP = typical_price
    S1 = PP - (range_1d * 1.1 / 12)
    S2 = PP - (range_1d * 1.1 / 6)
    S3 = PP - (range_1d * 1.1 / 4)
    R1 = PP + (range_1d * 1.1 / 12)
    R2 = PP + (range_1d * 1.1 / 6)
    R3 = PP + (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    # 1d volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / (vol_ma_20 + 1e-10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 1w data for trend filter (weekly bullish/bearish)
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(PP_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio_val = vol_ratio_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5  # Convert back to boolean
        
        if position == 0:
            # Long conditions: break above R3 or S3 with volume confirmation and bullish week
            if vol_ratio_val > 1.5 and weekly_bull:
                if price > R3_aligned[i] or price > S3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short conditions: break below S3 or R3 with volume confirmation and bearish week
            elif vol_ratio_val > 1.5 and not weekly_bull:
                if price < S3_aligned[i] or price < R3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to PP or breaks below S1 (reversal)
            if price <= PP_aligned[i] or price < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to PP or breaks above R1 (reversal)
            if price >= PP_aligned[i] or price > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals