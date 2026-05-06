#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Donchian breakout with volume confirmation and Choppiness regime filter.
# Long when price breaks above weekly Donchian high with volume > 1.5x average and Choppiness Index > 61.8 (range market).
# Short when price breaks below weekly Donchian low with volume > 1.5x average and Choppiness Index > 61.8.
# Weekly Donchian provides strong trend structure, volume confirms breakout strength, and Choppiness filter avoids trending markets where breakouts fail.
# Works in bull/bear markets by capturing genuine breakouts in ranging conditions.
# Target: 10-25 trades per year (40-100 over 4 years) with 0.25 position sizing.

name = "1d_1wDonchian_20_VolumeChop_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Donchian channels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate Choppiness Index (14-period) on daily timeframe
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first value to high-low to avoid NaN from roll
    tr[0] = high[0] - low[0]
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop = np.zeros(n)
    for i in range(n):
        if hl_range[i] > 0 and not np.isnan(tr_sum[i]):
            chop[i] = 100 * np.log10(tr_sum[i] / hl_range[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral value when range is zero
    
    # Choppiness filter: > 61.8 indicates ranging market (good for mean reversion/breakouts in range)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume and chop confirmation
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian low with volume and chop confirmation
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals