#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian breakout with volume confirmation.
# In trending markets (CHOP < 38.2), trade Donchian breakouts; in ranging markets (CHOP > 61.8), fade extremes.
# Uses 4h Donchian(20) breakout for trend entries and mean reversion at Bollinger Bands(20,2) in ranging markets.
# Volume > 1.3x 20-period average confirms participation. Designed for low trade frequency (<30/year) to avoid fee drag.
# Works in both bull and bear via regime adaptation.

name = "4h_Chop_Donchian_BB_MeanRev"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Initialize first ATR value
    if n > atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
    
    # Choppiness Index: 100 * log10(sum(atr/period) / (max(high)-min(low))) / log10(period)
    chop = np.full(n, 50.0)  # Default to neutral
    for i in range(2*atr_period, n):
        sum_atr = np.sum(tr[i-atr_period+1:i+1])
        max_high = np.max(high[i-atr_period+1:i+1])
        min_low = np.min(low[i-atr_period+1:i+1])
        if max_high > min_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(atr_period)
        else:
            chop[i] = 50.0
    
    # Donchian channels (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(donchian_period-1, n):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Bollinger Bands (20,2) for mean reversion in ranging markets
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_bb = np.full(n, np.nan)
    lower_bb = np.full(n, np.nan)
    for i in range(bb_period-1, n):
        sma[i] = np.mean(close[i-bb_period+1:i+1])
        std_dev[i] = np.std(close[i-bb_period+1:i+1])
        upper_bb[i] = sma[i] + bb_std * std_dev[i]
        lower_bb[i] = sma[i] - bb_std * std_dev[i]
    
    # Volume filter: > 1.3x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2*atr_period, donchian_period, bb_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market: CHOP < 38.2 -> trade Donchian breakouts
            if chop[i] < 38.2:
                if close[i] > upper_channel[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower_channel[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8 -> fade at Bollinger Bands
            elif chop[i] > 61.8:
                if close[i] < lower_bb[i] and volume_filter[i]:
                    signals[i] = 0.25  # Long at lower BB
                    position = 1
                elif close[i] > upper_bb[i] and volume_filter[i]:
                    signals[i] = -0.25  # Short at upper BB
                    position = -1
        elif position == 1:
            # Long exit: trend -> Donchian lower band; range -> SMA
            if chop[i] < 38.2:
                if close[i] < lower_channel[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging market
                if close[i] > sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: trend -> Donchian upper band; range -> SMA
            if chop[i] < 38.2:
                if close[i] > upper_channel[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging market
                if close[i] < sma[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals