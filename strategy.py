#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d volume spike + 1d chop regime filter
# Camarilla levels calculated from prior 1d OHLC. Long on break above R3 with volume spike in choppy market (CHOP>61.8).
# Short on break below S3 with volume spike in choppy market. Uses 1d timeframe for structure and regime,
# 12h for execution timing. Designed to work in both bull (breakouts) and bear (breakdowns) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1dVolSpike_Chop"
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
    
    # Get 1d data ONCE before loop for Camarilla, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) from prior day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Prior day values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    # Camarilla R3 and S3
    camarilla_r3 = prev_close + (1.1/12) * (prev_high - prev_low)
    camarilla_s3 = prev_close - (1.1/12) * (prev_high - prev_low)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume spike (volume > 1.5 * 20-period MA)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike_1d = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_spike_1d = np.zeros(len(vol_1d), dtype=bool)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(TR over period) / (max(high) - min(low))) / log10(period)
    if len(high_1d) >= 14:
        tr1 = np.abs(high_1d - low_1d)
        tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
        tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Sum of TR over 14 periods
        sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        # Max high and min low over 14 periods
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        # Avoid division by zero
        denominator = max_high - min_low
        denominator = np.where(denominator == 0, 1e-10, denominator)
        chop = 100 * np.log10(sum_tr / denominator) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
        # Chop regime: > 61.8 = ranging (good for mean reversion/breakouts in range)
        chop_regime = chop_aligned > 61.8
    else:
        chop_aligned = np.full(n, np.nan)
        chop_regime = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 50 to ensure sufficient warmup
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, volume spike, choppy market (CHOP > 61.8)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, volume spike, choppy market (CHOP > 61.8)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 (reversion to mean) or chop regime ends
            if close[i] < camarilla_s3_aligned[i] or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R3 (reversion to mean) or chop regime ends
            if close[i] > camarilla_r3_aligned[i] or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals