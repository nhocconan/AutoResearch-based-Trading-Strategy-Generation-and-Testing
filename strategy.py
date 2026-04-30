#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with Volume Confirmation
# Uses weekly Camarilla pivot levels (R3/S3, R4/S4) from 1w data as structure
# Breakout above R4 or below S4 with volume spike (2.0x 20-period average) signals continuation
# Pullback to R3/S3 with volume spike signals mean reversion in ranging markets
# Works in bull markets via buying R4 breakouts and selling R3 pullbacks
# Works in bear markets via selling S4 breakdowns and buying S3 pullbacks
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Weekly_Camarilla_Pivot_Breakout_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Based on previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3 = pivot + (range_1w * 1.1 / 4)
    s3 = pivot - (range_1w * 1.1 / 4)
    r4 = pivot + (range_1w * 1.1 / 2)
    s4 = pivot - (range_1w * 1.1 / 2)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above R4
                if curr_close > curr_r4:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price breaks below S4
                elif curr_close < curr_s4:
                    signals[i] = -0.25
                    position = -1
                # Bullish mean reversion: pullback to S3 in uptrend (price > pivot)
                elif curr_close >= curr_s3 and curr_close <= curr_r3 and curr_close > (r3_aligned[i] + s3_aligned[i])/2:
                    signals[i] = 0.25
                    position = 1
                # Bearish mean reversion: pullback to R3 in downtrend (price < pivot)
                elif curr_close >= curr_s3 and curr_close <= curr_r3 and curr_close < (r3_aligned[i] + s3_aligned[i])/2:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or R3 (failed mean reversion) or reaches R4 (take profit)
            if curr_close < curr_s3 or curr_close < curr_r3 or curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or S3 (failed mean reversion) or reaches S4 (take profit)
            if curr_close > curr_r3 or curr_close > curr_s3 or curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals