#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_VolumeRegime_v1
Breakout above/below 4h Donchian(20) channel with volume confirmation and 1d Choppiness regime filter.
Long when price breaks above upper band + volume spike + chop>61.8 (range).
Short when price breaks below lower band + volume spike + chop>61.8 (range).
Exit on opposite band touch or chop<38.2 (trend).
Designed for mean reversion in ranging markets, avoids whipsaw in strong trends.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Channel (20-period) ===
    # Upper band: highest high of last 20 periods
    highest_high = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
    
    # Lower band: lowest low of last 20 periods
    lowest_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # === Volume confirmation (20-period average) ===
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma[i] = volume[0]
    vol_spike = volume > vol_ma * 1.5  # 1.5x average volume
    
    # === 1d Choppiness Index (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros_like(close_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Sum of true range over 14 periods
    tr_sum = np.full_like(close_1d, np.nan)
    for i in range(len(tr_sum)):
        if i >= 13:
            tr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full_like(close_1d, np.nan)
    ll_14 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 13:
            hh_14[i] = np.max(high_1d[i-13:i+1])
            ll_14[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(tr_sum / (hh_14 - ll_14)) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    for i in range(len(chop)):
        if i >= 13 and hh_14[i] > ll_14[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Chop > 61.8 = ranging (good for mean reversion), Chop < 38.2 = trending
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    # Align chop to 4h timeframe
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike[i]) or 
            np.isnan(chop_ranging_aligned[i]) or 
            np.isnan(chop_trending_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions: only when flat
        if position == 0:
            # Long: break above upper band + volume spike + chop ranging
            if (close[i] > highest_high[i] and 
                vol_spike[i] and 
                chop_ranging_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower band + volume spike + chop ranging
            elif (close[i] < lowest_low[i] and 
                  vol_spike[i] and 
                  chop_ranging_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions
        elif position == 1:
            # Exit long: touch lower band OR chop becomes trending
            if (close[i] < lowest_low[i] or 
                chop_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch upper band OR chop becomes trending
            if (close[i] > highest_high[i] or 
                chop_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceChannel_Breakout_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0