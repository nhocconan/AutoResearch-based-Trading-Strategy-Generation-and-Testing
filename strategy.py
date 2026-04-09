#!/usr/bin/env python3
# 6h_camarilla_pivot_1d_volume_v1
# Hypothesis: 6h Camarilla pivot breakout/fade with 1d volume confirmation.
# Daily Camarilla levels (R3/S3, R4/S4) derived from prior 1d OHLC.
# Fade at R3/S3 (price reverses from extreme levels) with volume < 1.5x 20-period average.
# Breakout continuation at R4/S4 (price breaks extreme levels) with volume > 1.8x 20-period average.
# Uses 6h timeframe for entries, 1d HTF for pivot levels and volume filter.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volume_v1"
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
    
    # Daily HTF data for Camarilla pivots and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R4, R3, S3, S4
    range_ = prev_high - prev_low
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily volume confirmation filter
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S3 (fade failed) OR breaks above R4 (take profit)
            if close[i] < camarilla_s3_aligned[i] or close[i] > camarilla_r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above R3 (fade failed) OR breaks below S4 (take profit)
            if close[i] > camarilla_r3_aligned[i] or close[i] < camarilla_s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume filters: low volume for fade, high volume for breakout
            low_volume = volume[i] < 1.5 * volume_ma_1d_aligned[i]
            high_volume = volume[i] > 1.8 * volume_ma_1d_aligned[i]
            
            # Fade at R3/S3: price touches extreme level with low volume (likely reversal)
            if abs(high[i] - camarilla_r3_aligned[i]) < 0.001 * camarilla_r3_aligned[i] and low_volume:
                # Price touched R3 from below, expect reversal down
                if close[i] < camarilla_r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif abs(low[i] - camarilla_s3_aligned[i]) < 0.001 * camarilla_s3_aligned[i] and low_volume:
                # Price touched S3 from above, expect reversal up
                if close[i] > camarilla_s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
            # Breakout at R4/S4: price breaks extreme level with high volume (continuation)
            elif close[i] > camarilla_r4_aligned[i] and high_volume:
                position = 1
                signals[i] = 0.25
            elif close[i] < camarilla_s4_aligned[i] and high_volume:
                position = -1
                signals[i] = -0.25
    
    return signals