#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Uses daily Camarilla pivot levels (R1, S1, R2, S2, R3, S3, R4, S4)
# - Long when price breaks above R1 with volume confirmation, short when breaks below S1
# - Continuation signals at R4/S4 breaks, reversals at R3/S3
# - Volume filter: current 6h volume > 1.5x average daily volume per 6h bar
# - Designed to capture intraday momentum in both trending and ranging markets
# - Target: 15-25 trades/year per symbol to minimize fee drag

name = "6h_Camarilla_R1S1_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: R4 = C + ((H-L)*1.1/2)*1.1, R3 = C + ((H-L)*1.1/2)*1.1/2, etc.
    # Actually: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Where C = (H+L+Close)/3 (typical price)
    
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_range = df_1d['high'] - df_1d['low']
    
    # Calculate pivot levels
    camarilla_c = typical_price.values
    camarilla_r1 = camarilla_c + (daily_range * 1.1 / 12)
    camarilla_s1 = camarilla_c - (daily_range * 1.1 / 12)
    camarilla_r2 = camarilla_c + (daily_range * 1.1 / 6)
    camarilla_s2 = camarilla_c - (daily_range * 1.1 / 6)
    camarilla_r3 = camarilla_c + (daily_range * 1.1 / 4)
    camarilla_s3 = camarilla_c - (daily_range * 1.1 / 4)
    camarilla_r4 = camarilla_c + (daily_range * 1.1 / 2)
    camarilla_s4 = camarilla_c - (daily_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: average daily volume per 6h bar
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    # 1 day = 4 six-hour bars, so divide by 4 to get average volume per 6h bar
    vol_ma_6h = vol_ma_1d / 4.0
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x average 6h volume
        volume_filter = vol_ma_6h_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_6h_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above R1 with volume
            if close[i] > camarilla_r1_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S1 with volume
            elif close[i] < camarilla_s1_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position management
            # Exit conditions:
            # 1. Price reaches R4 (take profit)
            # 2. Price falls back below S1 (stop/reversal)
            # 3. Price breaks below R3 (potential reversal)
            if (close[i] >= camarilla_r4_aligned[i] or 
                close[i] <= camarilla_s1_aligned[i] or
                close[i] < camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position management
            # Exit conditions:
            # 1. Price reaches S4 (take profit)
            # 2. Price rises back above R1 (stop/reversal)
            # 3. Price breaks above S3 (potential reversal)
            if (close[i] <= camarilla_s4_aligned[i] or 
                close[i] >= camarilla_r1_aligned[i] or
                close[i] > camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals