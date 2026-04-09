#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v3
# Hypothesis: 6h strategy using weekly Camarilla pivot levels (from 1w HTF) for structure and 6h Donchian breakout for entry.
# Long: Price breaks above 6h Donchian(20) high AND price > weekly R3 pivot level AND volume > 1.5x 20-period average.
# Short: Price breaks below 6h Donchian(20) low AND price < weekly S3 pivot level AND volume > 1.5x 20-period average.
# Exit: Opposite Donchian breakout or price crosses weekly pivot center (R4/S4 level).
# Uses weekly pivots for higher timeframe structure to avoid counter-trend trades in ranging markets.
# Volume confirmation filters weak breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 6h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Camarilla pivot levels (1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla pivot formulas
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    range_ = weekly_high - weekly_low
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s4 = pivot - (range_ * 1.1)
    
    # Align weekly levels to 6h timeframe (with 1-bar delay for completed weekly bar)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR crosses weekly S4 level
            if close[i] < donchian_low[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR crosses weekly R4 level
            if close[i] > donchian_high[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, above weekly R3, volume confirmed
            if (close[i] > donchian_high[i] and close[i] > r3_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, below weekly S3, volume confirmed
            elif (close[i] < donchian_low[i] and close[i] < s3_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals