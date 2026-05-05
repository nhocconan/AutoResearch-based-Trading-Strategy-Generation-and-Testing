#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and chop regime filter
# Long when price breaks above weekly Donchian high(20) AND volume > 1.5 * avg_volume(50) AND chop(14) < 61.8
# Short when price breaks below weekly Donchian low(20) AND volume > 1.5 * avg_volume(50) AND chop(14) < 61.8
# Exit when price crosses back below/above weekly Donchian midpoint OR chop(14) > 61.8 (range regime)
# Uses discrete sizing 0.25 to balance return and risk
# Weekly Donchian provides robust structure from higher timeframe
# Volume confirms breakout strength
# Chop filter avoids whipsaws in ranging markets
# Works in bull markets (breakouts with uptrend structure) and bear markets (breakdowns with downtrend structure)

name = "1d_WeeklyDonchian20_Breakout_Volume_ChopFilter"
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
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Midpoint = (Upper + Lower)/2
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian levels to 1d timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get 1d data ONCE before loop for volume and chop calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for volume MA and chop
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate volume confirmation: volume > 1.5 * 50-period average volume on 1d
    avg_volume_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume_1d > (1.5 * avg_volume_50)
    
    # Calculate Choppiness Index (14) on 1d
    # CHOP = 100 * log10(sum(ATR(1), 14) / (log10(14) * (max(high,14) - min(low,14))))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(n) * range)) / log10(n)
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    log10_14 = np.log10(14)
    chop = 100 * (np.log10(atr1 + 1e-10) - np.log10(log10_14 * range_14 + 1e-10)) / log10_14
    chop_filter = chop < 61.8  # Trending regime (CHOP < 61.8)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume_50[i]) or 
            np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper, volume confirmation, trending regime, in session
            if close[i] > donchian_upper_aligned[i] and volume_confirm[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower, volume confirmation, trending regime, in session
            elif close[i] < donchian_lower_aligned[i] and volume_confirm[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian midpoint OR chop > 61.8 (range regime)
            if close[i] < donchian_mid_aligned[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian midpoint OR chop > 61.8 (range regime)
            if close[i] > donchian_mid_aligned[i] or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals