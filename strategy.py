#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1d HMA21 trend filter and volume spike confirmation
# Long when price breaks above weekly Donchian upper (20) AND price > 1d HMA21 AND volume > 1.5 * avg_volume(20) on 1d
# Short when price breaks below weekly Donchian lower (20) AND price < 1d HMA21 AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price crosses back below/above weekly Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 50-120 total trades over 4 years (12-30/year) for 1d timeframe
# Weekly Donchian provides robust structure from higher timeframe
# 1d HMA21 filters primary trend with reduced lag vs EMA/SMA
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "1d_Donchian20_HMA21_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Mid = (Upper + Lower)/2
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian levels to 1d timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need enough for HMA21
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA21 (Hull Moving Average)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    n_half = 21 // 2
    n_sqrt = int(np.sqrt(21))
    
    wma_half = np.array([np.nan] * len(close_1d))
    wma_full = np.array([np.nan] * len(close_1d))
    
    for i in range(n_half, len(close_1d)):
        wma_half[i] = wma(close_1d[i-n_half+1:i+1], n_half)
    for i in range(21, len(close_1d)):
        wma_full[i] = wma(close_1d[i-21+1:i+1], 21)
    
    raw_hma = 2 * wma_half - wma_full
    hma21_1d = np.array([np.nan] * len(close_1d))
    for i in range(n_sqrt, len(raw_hma)):
        if not np.isnan(raw_hma[i]):
            hma21_1d[i] = wma(raw_hma[i-n_sqrt+1:i+1], n_sqrt)
    
    hma21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma21_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(hma21_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper, above 1d HMA21, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and close[i] > hma21_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly Donchian lower, below 1d HMA21, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and close[i] < hma21_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals