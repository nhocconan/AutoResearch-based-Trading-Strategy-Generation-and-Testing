#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
# Donchian(20) breakout captures momentum in both bull/bear markets
# 1d volume spike (>1.5x 20-period average) confirms institutional participation
# Choppiness Index (14) > 61.8 = ranging (fade breakouts), < 38.2 = trending (follow breakouts)
# Position size 0.25 to limit drawdown (BTC 2022 drawdown manageable)
# Target: 100-180 total trades over 4 years (25-45/year) for optimal fee/alpha balance
# Works in bull via breakout continuation, in bear via chop-filtered mean reversion at extremes

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-20:i])
    
    # Calculate 1d Choppiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # Sum of TR over 14 periods
    tr_sum_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        tr_sum_14[i] = np.sum(tr_1d[i-14:i])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full(len(df_1d), np.nan)
    ll_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        hh_14[i] = np.max(high_1d[i-14:i])
        ll_14[i] = np.min(low_1d[i-14:i])
    
    # Chop = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if hh_14[i] > ll_14[i]:
            chop_1d[i] = 100 * np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Align 1d data to 4h timeframe
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_1d_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20_4h[i]) or 
            np.isnan(chop_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = vol_ma_20_4h[i]
        chop = chop_1d_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop > 61.8:  # Ranging regime - exit at mean reversion
                if close[i] <= (highest_high[i] + lowest_low[i]) / 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Trending regime - exit on Donchian reversal
                if close[i] < lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop > 61.8:  # Ranging regime - exit at mean reversion
                if close[i] >= (highest_high[i] + lowest_low[i]) / 2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Trending regime - exit on Donchian reversal
                if close[i] > highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if vol_ma > 0 and volume[i] > 1.5 * vol_ma:  # Volume confirmation
                if chop < 38.2:  # Trending regime - follow breakout
                    if close[i] > highest_high[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop > 61.8:  # Ranging regime - fade extremes
                    if close[i] < lowest_low[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] > highest_high[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals