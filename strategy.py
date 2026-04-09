#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and chop regime filter
# Uses 4h Donchian(20) breakouts for trend capture
# Confirms with 1d volume > 1.8x 20-day average (institutional participation)
# Uses 1d Choppiness Index > 61.8 for ranging markets (avoid false breakouts in chop)
# Exits when price closes opposite Donchian level or chop regime shifts
# Position size 0.25 to limit drawdown
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag
# Works in both bull/bear: Donchian captures trends, chop filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    vol_sum = 0.0
    for i in range(len(df_1d)):
        vol_sum += vol_1d[i]
        if i >= 20:
            vol_sum -= vol_1d[i-20]
        if i >= 19:
            vol_ma_20_1d[i] = vol_sum / 20
    
    # Calculate 1d Choppiness Index (14-period)
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
    tr_sum = 0.0
    for i in range(len(df_1d)):
        tr_sum += tr_1d[i]
        if i >= 14:
            tr_sum -= tr_1d[i-14]
        if i >= 13:
            tr_sum_14[i] = tr_sum
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full(len(df_1d), np.nan)
    ll_14 = np.full(len(df_1d), np.nan)
    hh = high_1d[0]
    ll = low_1d[0]
    for i in range(len(df_1d)):
        if high_1d[i] > hh:
            hh = high_1d[i]
        if low_1d[i] < ll:
            ll = low_1d[i]
        if i >= 14:
            if high_1d[i-14] == hh:
                hh = max(high_1d[i-13:i+1]) if i+1 <= len(df_1d) else max(high_1d[i-13:])
            if low_1d[i-14] == ll:
                ll = min(low_1d[i-13:i+1]) if i+1 <= len(df_1d) else min(low_1d[i-13:])
        if i >= 13:
            hh_14[i] = hh
            ll_14[i] = ll
    
    # Choppiness Index
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if tr_sum_14[i] > 0 and hh_14[i] > ll_14[i]:
            chop_1d[i] = 100 * np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Align 1d data to 4h timeframe (only use completed daily bars)
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    hh_4h = high[0]
    ll_4h = low[0]
    for i in range(n):
        if high[i] > hh_4h:
            hh_4h = high[i]
        if low[i] < ll_4h:
            ll_4h = low[i]
        if i >= 20:
            if high[i-20] == hh_4h:
                hh_4h = max(high[i-19:i+1]) if i+1 <= n else max(high[i-19:])
            if low[i-20] == ll_4h:
                ll_4h = min(low[i-19:i+1]) if i+1 <= n else min(low[i-19:])
        if i >= 19:
            donchian_h[i] = hh_4h
            donchian_l[i] = ll_4h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or 
            np.isnan(vol_ma_20_4h[i]) or 
            np.isnan(chop_4h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging to trending markets (avoid extreme chop > 61.8)
        if chop_4h[i] > 61.8:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low
            if close[i] <= donchian_l[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high
            if close[i] >= donchian_h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current 4h volume > 1.8x 1d average volume per 4h bar
            # Approximate: 1d volume / 6 = average 4h bar volume
            vol_4h_avg_estimate = vol_ma_20_4h[i] / 6.0
            vol_ratio = volume[i] / vol_4h_avg_estimate if vol_4h_avg_estimate > 0 else 0
            
            # Enter long: price closes above 4h Donchian high with volume confirmation
            if (close[i] > donchian_h[i] and 
                vol_ratio > 1.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian low with volume confirmation
            elif (close[i] < donchian_l[i] and 
                  vol_ratio > 1.8):
                position = -1
                signals[i] = -0.25
    
    return signals