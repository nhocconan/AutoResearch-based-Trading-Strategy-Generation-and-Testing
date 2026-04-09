#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d/1w Camarilla pivot levels with volume confirmation and chop regime filter
# Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) identify institutional support/resistance
# Mean reversion at R3/S3 when choppy market (CHOP > 61.8), breakout continuation at R4/S4 when trending (CHOP < 38.2)
# Volume confirmation ensures institutional participation
# Fixed position size 0.25 to balance return and drawdown
# Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)

name = "6h_1d_1w_camarilla_chop_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 5 or len(df_1w) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    range_1d = high_1d[-1] - low_1d[-1]
    r3_1d = pivot_1d + range_1d * 1.1 / 2
    s3_1d = pivot_1d - range_1d * 1.1 / 2
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Calculate 1w Camarilla pivot levels
    pivot_1w = (high_1w[-1] + low_1w[-1] + close_1w[-1]) / 3.0
    range_1w = high_1w[-1] - low_1w[-1]
    r3_1w = pivot_1w + range_1w * 1.1 / 2
    s3_1w = pivot_1w - range_1w * 1.1 / 2
    r4_1w = pivot_1w + range_1w * 1.1
    s4_1w = pivot_1w - range_1w * 1.1
    
    # Align 1d Camarilla levels to 6h timeframe (constant until new 1d bar)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, pivot_1d))
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, r3_1d))
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, s3_1d))
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, r4_1d))
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, s4_1d))
    
    # Align 1w Camarilla levels to 6h timeframe (constant until new 1w bar)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, pivot_1w))
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, r3_1w))
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, s3_1w))
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, r4_1w))
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, s4_1w))
    
    # Calculate 1d Choppiness Index (CHOP) for regime detection
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    sum_atr = pd.Series(atr_1d).rolling(window=atr_period, min_periods=atr_period).sum().values
    range_1d_period = highest_high_1d - lowest_low_1d
    chop_1d = 100 * np.log10(sum_atr / range_1d_period) / np.log10(atr_period)
    chop_1d = np.where(range_1d_period == 0, 50, chop_1d)  # Avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute volume confirmation (24-period average for 6h, approx 6d)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_24[i]) or
            vol_ma_24[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_24[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion to pivot in choppy, or stop at S3/S4
            if chop_1d_aligned[i] > 61.8:  # Choppy regime - mean reversion
                if close[i] < pivot_1d_aligned[i] or close[i] < pivot_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # Trending regime - trail with S3/S4
                if close[i] < s3_1d_aligned[i] or close[i] < s3_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions: mean reversion to pivot in choppy, or stop at R3/R4
            if chop_1d_aligned[i] > 61.8:  # Choppy regime - mean reversion
                if close[i] > pivot_1d_aligned[i] or close[i] > pivot_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # Trending regime - trail with R3/R4
                if close[i] > r3_1d_aligned[i] or close[i] > r3_1w_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on regime
            if chop_1d_aligned[i] > 61.8:  # Choppy regime - mean reversion at R3/S3
                # Long at S3 rejection
                if close[i] <= s3_1d_aligned[i] * 1.005 and close[i] > s3_1d_aligned[i]:
                    # Additional confirmation: price also above weekly S3
                    if close[i] > s3_1w_aligned[i]:
                        position = 1
                        signals[i] = position_size
                # Short at R3 rejection
                elif close[i] >= r3_1d_aligned[i] * 0.995 and close[i] < r3_1d_aligned[i]:
                    # Additional confirmation: price also below weekly R3
                    if close[i] < r3_1w_aligned[i]:
                        position = -1
                        signals[i] = -position_size
            else:  # Trending regime - breakout continuation at R4/S4
                # Long on R4 breakout
                if close[i] > r4_1d_aligned[i] * 1.002 or close[i] > r4_1w_aligned[i] * 1.002:
                    position = 1
                    signals[i] = position_size
                # Short on S4 breakdown
                elif close[i] < s4_1d_aligned[i] * 0.998 or close[i] < s4_1w_aligned[i] * 0.998:
                    position = -1
                    signals[i] = -position_size
    
    return signals