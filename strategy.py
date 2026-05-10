#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (12) crossing zero with volume spike confirms momentum, filtered by Choppiness Index regime (CHOP > 61.8 = range, < 38.2 = trend). In trend regime, follow TRIX cross; in range, fade extremes. Uses 1d trend filter to avoid counter-trend trades. Designed for low trade frequency (<50/year) to minimize fee drag. Works in bull/bear via regime adaptation.
"""

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # TRIX(12,9,9) - triple EMA then % change
    ema1 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix = trix.fillna(0).values
    
    # Volume spike (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14) - range/trend detector
    atr1 = pd.Series(np.maximum(high_s - low_s,
                                np.maximum(high_s - close_s.shift(1),
                                           close_s.shift(1) - low_s))).rolling(window=14, min_periods=14).mean().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1.sum() / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div/0
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        # Regime filters
        chop_value = chop[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        if position == 0:
            # Long entry: TRIX crosses above zero + volume + trend regime + 1d uptrend
            if (trix[i] > 0 and trix[i-1] <= 0 and volume_confirm and
                is_trending and trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero + volume + trend regime + 1d downtrend
            elif (trix[i] < 0 and trix[i-1] >= 0 and volume_confirm and
                  is_trending and trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            # Range fade: sell at resistance in range
            elif (is_ranging and close[i] >= highest_high[i] and volume_confirm):
                signals[i] = -0.20
                position = -1
            # Range fade: buy at support in range
            elif (is_ranging and close[i] <= lowest_low[i] and volume_confirm):
                signals[i] = 0.20
                position = 1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero OR chop turns ranging
            if (trix[i] < 0 or (is_ranging and close[i] >= highest_high[i] * 0.98)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero OR chop turns ranging
            if (trix[i] > 0 or (is_ranging and close[i] <= lowest_low[i] * 1.02)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals