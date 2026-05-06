#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Choppiness Index regime filter + 1w Donchian breakout
# Uses Choppiness Index (14) on 6h to filter ranging vs trending markets
# In trending markets (CHOP < 38.2): trade breakouts of 1w Donchian channels
# In ranging markets (CHOP > 61.8): fade at 1w Bollinger Bands (2.5 std)
# Volume confirmation (>1.3x 20-bar average) ensures institutional participation
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: trend following in trends, mean reversion in ranges

name = "6h_ChopRegime_1wDonchianBB_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_1w) < 20 or len(df_6h) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Choppiness Index (14)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = np.abs(high_6h[1:] - low_6h[1:])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
    range_14 = max_high_14 - min_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    
    # Calculate 1w Donchian channels (20)
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w Bollinger Bands (20, 2.5)
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.5 * std_20
    bb_lower = sma_20 - 2.5 * std_20
    
    # Calculate volume confirmation filter (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_6h, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market: CHOP < 38.2
            if chop_aligned[i] < 38.2:
                # Long breakout: price breaks above Donchian high
                if close[i] > donch_high_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: price breaks below Donchian low
                elif close[i] < donch_low_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8
            elif chop_aligned[i] > 61.8:
                # Long at lower Bollinger Band
                if close[i] < bb_lower_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at upper Bollinger Band
                elif close[i] > bb_upper_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below midline or opposite signal
            midline = (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2
            if close[i] < midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midline or opposite signal
            midline = (bb_upper_aligned[i] + bb_lower_aligned[i]) / 2
            if close[i] > midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals