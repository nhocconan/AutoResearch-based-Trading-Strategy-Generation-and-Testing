#!/usr/bin/env python3
# 12h_donchian20_volume_regime_v1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and chop regime filter.
# Long when price breaks above upper band with volume > 1.5x average and chop > 61.8 (range).
# Short when price breaks below lower band with volume > 1.5x average and chop > 61.8.
# Exit when price crosses opposite band or volume drops below average.
# Uses 1d trend filter (price > SMA50 for long, < SMA50 for short) to avoid counter-trend.
# Target: 15-30 trades/year to stay under 120 total over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donch_len-1, n):
        upper[i] = np.max(high[i-donch_len+1:i+1])
        lower[i] = np.min(low[i-donch_len+1:i+1])
    
    # Volume filter: 1.5x 20-period average
    vol_ma_len = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_len-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_len+1:i+1])
    vol_surge = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Chop regime filter (14-period) - range when > 61.8
    chop_len = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = np.full(n, np.nan)
    for i in range(chop_len-1, n):
        atr[i] = np.mean(tr[i-chop_len+1:i+1])
    highest = np.maximum.accumulate(high)
    lowest = np.minimum.accumulate(low)
    chop = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0 and highest[i] > lowest[i]:
            chop[i] = 100 * np.log10((highest[i] - lowest[i]) / (atr[i] * chop_len)) / np.log10(chop_len)
    chop_range = chop > 61.8  # ranging market
    
    # 1d trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_len, vol_ma_len, chop_len, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i]) or np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below lower band or volume drops below average
            if close[i] < lower[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above upper band or volume drops below average
            if close[i] > upper[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper band, volume surge, chop range, 1d uptrend
            if (close[i] > upper[i] and 
                vol_surge[i] and 
                chop_range[i] and 
                close[i] > sma50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower band, volume surge, chop range, 1d downtrend
            elif (close[i] < lower[i] and 
                  vol_surge[i] and 
                  chop_range[i] and 
                  close[i] < sma50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals