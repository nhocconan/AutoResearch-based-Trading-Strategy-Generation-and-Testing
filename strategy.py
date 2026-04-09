#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot long/short entries from 1d levels + volume spike + chop regime filter
# Camarilla pivot levels act as intraday support/resistance; price touching S3/R3 with volume confirms institutional interest
# Choppiness filter avoids whipsaws in ranging markets; discrete sizing 0.30 controls drawdown
# Works in bull/bear: pivot levels adapt to volatility, volume confirms breakout authenticity
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (S1,S2,S3,R1,R2,R3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    range_1d = high_1d - low_1d
    s1 = close_1d - (range_1d * 1.1 / 12)
    s2 = close_1d - (range_1d * 1.1 / 6)
    s3 = close_1d - (range_1d * 1.1 / 4)
    r1 = close_1d + (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for 1d bar close)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Calculate 4h choppiness index (14-period) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    atr14 = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            atr14[i] = np.nan
        else:
            atr14[i] = np.mean(tr[i-14:i])
    
    sum_tr14 = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            sum_tr14[i] = np.nan
        else:
            sum_tr14[i] = np.sum(tr[i-14:i])
    
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14 or np.isnan(sum_tr14[i]) or sum_tr14[i] == 0:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(sum_tr14[i] / (atr14[i] * 14)) / np.log10(10)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop[i]) or np.isnan(avg_volume[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        # Chop regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        chop_ranging = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < S1 OR chop becomes too low (trending strong)
            if close[i] < s1_1d_aligned[i] or chop[i] < 40.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price > R1 OR chop becomes too low (trending strong)
            if close[i] > r1_1d_aligned[i] or chop[i] < 40.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Entry logic with volume confirmation and chop regime
            if volume_confirmed and chop_ranging:
                # Long entry: price < S3 AND price > S2 (deep oversold in range)
                if close[i] < s3_1d_aligned[i] and close[i] > s2_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price > R3 AND price < R2 (deep overbought in range)
                elif close[i] > r3_1d_aligned[i] and close[i] < r2_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals