#!/usr/bin/env python3
# 4h_AntiFragile_Breakout_With_Volume_Regime_Filter
# Hypothesis: Combine Donchian(20) breakout with volume confirmation and Choppiness Index regime filter.
# Uses Choppiness Index > 61.8 for range (mean-revert) and < 38.2 for trend (follow breakout).
# In trending regimes: breakout entries. In ranging regimes: fade extremes at Bollinger Bands.
# Designed for low trade frequency (<40/year) to minimize fee drag and work in both bull/bear markets.

name = "4h_AntiFragile_Breakout_With_Volume_Regime_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Bollinger Bands (20, 2) for mean reversion in ranging markets
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Choppiness Index (14-period) for regime detection
    def choppiness_index(high, low, close, window=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # ATR (smoothed TR)
        atr = np.full_like(close, np.nan)
        for i in range(window, len(atr)):
            atr[i] = np.mean(tr[i - window + 1:i + 1])
        
        # Sum of TR over window
        sum_tr = np.full_like(close, np.nan)
        for i in range(window - 1, len(sum_tr)):
            sum_tr[i] = np.sum(tr[i - window + 1:i + 1])
        
        # Highest high and lowest low over window
        highest_high = rolling_max(high, window)
        lowest_low = rolling_min(low, window)
        
        # Chop formula: 100 * log10(sum_tr / (highest_high - lowest_low)) / log10(window)
        range_val = highest_high - lowest_low
        chop = np.full_like(close, np.nan)
        mask = (sum_tr > 0) & (range_val > 0) & ~np.isnan(range_val)
        chop[mask] = 100 * np.log10(sum_tr[mask] / range_val[mask]) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # Volume confirmation: 20-period average volume
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or \
           np.isnan(chop[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        # Neutral zone (38.2-61.8) defaults to no action to avoid whipsaw
        
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            if is_trending and volume_condition:
                # Trend following: breakout entries
                if close[i] > donchian_high[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and volume_condition:
                # Mean reversion: fade Bollinger Bands extremes
                if close[i] > bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
                elif close[i] < bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit on breakdown or range entry
                if close[i] < donchian_low[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # Exit at mean reversion
                if close[i] > bb_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit on breakout or range entry
                if close[i] > donchian_high[i] or chop[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # Exit at mean reversion
                if close[i] < bb_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals