#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot breakout with volume confirmation and choppiness regime filter
# Long when price breaks above 1w Camarilla R3 level AND volume > 1.5 * avg_volume(20) AND choppiness < 61.8 (trending)
# Short when price breaks below 1w Camarilla S3 level AND volume > 1.5 * avg_volume(20) AND choppiness < 61.8 (trending)
# Exit when price crosses back below/above 1w Camarilla pivot point
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# 1w Camarilla provides robust weekly support/resistance from higher timeframe
# Volume confirmation ensures breakout strength
# Choppiness filter avoids false breakouts in ranging markets
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "1d_Camarilla_R3S3_Breakout_Volume_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one completed 1w bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (based on previous 1w bar)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + Range * 1.1/2, S3 = Pivot - Range * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    camarilla_r3 = pivot_1w + (range_1w * 1.1 / 2.0)
    camarilla_s3 = pivot_1w - (range_1w * 1.1 / 2.0)
    camarilla_pivot = pivot_1w  # PP level for exit
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First TR
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop_range = hh14 - ll14
    # Avoid division by zero
    chop_range_safe = np.where(chop_range == 0, 1e-10, chop_range)
    chop = 100 * np.log10(chop_sum / chop_range_safe) / np.log10(14)
    chop_filter = chop < 61.8  # Trending regime (chop < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 1w Camarilla R3, volume confirmation, trending regime
            if close[i] > camarilla_r3_aligned[i] and volume_confirm[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1w Camarilla S3, volume confirmation, trending regime
            elif close[i] < camarilla_s3_aligned[i] and volume_confirm[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 1w Camarilla pivot
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 1w Camarilla pivot
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals