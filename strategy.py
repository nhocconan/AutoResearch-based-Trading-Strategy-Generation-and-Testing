#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with Daily Volume Spike and Chop Regime Filter
# Camarilla R3/S3 levels act as strong intraday support/resistance on 12h chart
# Breakouts above R3 or below S3 with volume confirmation indicate institutional participation
# Daily chop filter (BB Width percentile) avoids false breakouts in ranging markets
# Works in bull markets (buying breakouts) and bear markets (selling breakdowns)
# Target: 12-37 trades/year (50-150 total over 4 years)

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R3, S3) from previous day
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # Simplified: range = high - low, R3 = close + range*1.1/2, S3 = close - range*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for previous day (to avoid look-ahead)
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align daily pivot levels to 12h timeframe (shifted by 1 day for completed bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily volume spike: volume > 2.0x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Daily chop filter: BB Width > 50th percentile = ranging market (avoid breakouts)
    bb_middle_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_width_1d = np.where(bb_middle_1d != 0, (bb_middle_1d + 2*bb_std_1d - (bb_middle_1d - 2*bb_std_1d)) / bb_middle_1d, 0)
    bb_width_series = pd.Series(bb_width_1d)
    chop_threshold = bb_width_series.rolling(window=50, min_periods=50).quantile(0.50).values  # 50th percentile
    chop_condition = bb_width_1d > chop_threshold  # True = choppy/ranging market
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_condition)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 50, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike_aligned[i]
        curr_chop = chop_aligned[i]
        
        # Only trade in non-choppy markets (trending regime)
        if curr_chop:
            # In choppy markets, stay flat or exit
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above R3 with volume spike
            if curr_close > curr_r3 and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with volume spike
            elif curr_close < curr_s3 and curr_volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when price returns to midpoint
            # Exit when price crosses below the pivot point (midpoint of R3/S3)
            pivot_point = (curr_r3 + curr_s3) / 2
            if curr_close < pivot_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price returns to midpoint
            # Exit when price crosses above the pivot point
            pivot_point = (curr_r3 + curr_s3) / 2
            if curr_close > pivot_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals