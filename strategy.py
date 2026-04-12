#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R reversal with 1d volume spike and chop regime filter
    # Williams %R identifies overbought/oversold conditions for mean reversion
    # Volume spike confirms institutional participation at extremes
    # Chop regime filter ensures mean reversion only in ranging markets
    # Works in bull/bear by fading extremes in range and avoiding false signals in trends
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    williams_r = np.full(len(df_1d), np.nan)
    highest_high = np.full(len(df_1d), np.nan)
    lowest_low = np.full(len(df_1d), np.nan)
    
    for i in range(13, len(df_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Align 1d Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d volume spike filter (current volume > 1.5 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 1.5 * vol_ma_20_1d_aligned
    
    # Chop regime filter (using 1d data)
    # Chop > 61.8 = ranging (mean revert at extremes)
    # Chop < 38.2 = trending (avoid mean reversion in strong trends)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        tr = np.max([
            high_1d[i] - low_1d[i],
            np.abs(high_1d[i] - close_1d[i-1]),
            np.abs(low_1d[i] - close_1d[i-1])
        ])
        if i == 14:
            atr_1d[i] = np.mean([high_1d[j] - low_1d[j] for j in range(i-13, i+1)])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    # Calculate Chop (14-period)
    sum_tr_14 = np.full(len(df_1d), np.nan)
    max_min_range_14 = np.full(len(df_1d), np.nan)
    chop = np.full(len(df_1d), 50.0)  # default neutral
    
    for i in range(13, len(df_1d)):
        # Sum of true range over 14 periods
        tr_sum = 0
        for j in range(i-13, i+1):
            tr = np.max([
                high_1d[j] - low_1d[j],
                np.abs(high_1d[j] - close_1d[j-1]) if j > 0 else 0,
                np.abs(low_1d[j] - close_1d[j-1]) if j > 0 else 0
            ])
            tr_sum += tr
        sum_tr_14[i] = tr_sum
        
        # Max high - min low over 14 periods
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        max_min_range_14[i] = max_high - min_low
        
        if max_min_range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i] / max_min_range_14[i]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (Chop > 61.8)
        if chop_aligned[i] > 61.8:
            # Williams %R extremes for mean reversion
            long_entry = williams_r_aligned[i] <= -80 and volume_spike[i]  # oversold
            short_entry = williams_r_aligned[i] >= -20 and volume_spike[i]  # overbought
            long_exit = williams_r_aligned[i] >= -50  # exit at midpoint
            short_exit = williams_r_aligned[i] <= -50  # exit at midpoint
        else:
            # In trending markets, stay flat to avoid whipsaws
            long_entry = False
            short_entry = False
            long_exit = True
            short_exit = True
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williamsr_vol_chop_v1"
timeframe = "12h"
leverage = 1.0