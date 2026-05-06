#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot R3/S3 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R3 level AND volume > 2.0 * avg_volume(20) AND CHOP(14) > 61.8 (range regime)
# Short when price breaks below 1d Camarilla S3 level AND volume > 2.0 * avg_volume(20) AND CHOP(14) > 61.8 (range regime)
# Exit when price retests the 1d Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla pivots provide strong intraday support/resistance levels with high reversal probability
# Volume confirmation validates breakout strength while limiting false signals
# Choppiness filter ensures we only trade in ranging markets where mean reversion works best
# Works in both bull (buy R3 breakouts) and bear (sell S3 breakdowns) markets by fading extremes

name = "12h_CamarillaR3S3_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 completed daily bars for pivot calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R3 = Close + (High - Low) * 1.1 / 4
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    # S3 = Close - (High - Low) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Calculate Choppiness Index (CHOP) on 12h timeframe
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(period) * range))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero and log of zero
    log_range = np.log10(np.maximum(range_14, 1e-10))
    log_atr_sum = np.log10(np.maximum(atr_14 * 14, 1e-10))
    log_n = np.log10(14)
    chop = 100 * (log_atr_sum / (log_n * log_range))
    # Handle edge cases where range is zero ( chop becomes 100 * log(small)/0 -> inf)
    chop = np.where(range_14 == 0, 100.0, chop)
    chop_regime = chop > 61.8  # Range regime (mean revert)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, volume spike, in range regime
            if (close[i] > r3_aligned[i] and 
                volume_confirm[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, volume spike, in range regime
            elif (close[i] < s3_aligned[i] and 
                  volume_confirm[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 1d Camarilla pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 1d Camarilla pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals