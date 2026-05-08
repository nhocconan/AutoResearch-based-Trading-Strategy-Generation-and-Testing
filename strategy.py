# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout + Daily Volume Spike + Choppiness Regime
# Uses daily Camarilla pivot levels for structure, daily volume spike (>2x 20-day avg) for momentum confirmation,
# and daily choppiness index to filter choppy markets. Designed to work in both bull and bear markets
# by following price action around institutional pivot levels while avoiding low-momentum chop.
# Target: 15-30 trades/year on 12h timeframe.

name = "12h_Camarilla_Pivot_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots, volume average, and choppiness
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close = np.roll(close_daily, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R4 = pivot + range_val * 1.5
    R3 = pivot + range_val * 1.25
    R2 = pivot + range_val * 1.166
    R1 = pivot + range_val * 1.083
    S1 = pivot - range_val * 1.083
    S2 = pivot - range_val * 1.166
    S3 = pivot - range_val * 1.25
    S4 = pivot - range_val * 1.5
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Calculate daily choppiness index (14-period)
    atr_14_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        tr = np.maximum(high_daily[1:] - low_daily[1:], 
                        np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                                   np.abs(low_daily[1:] - close_daily[:-1])))
        tr = np.concatenate([[np.nan], tr])
        for i in range(14, len(tr)):
            if np.isnan(atr_14_daily[i-1]):
                atr_14_daily[i] = np.nanmean(tr[i-13:i+1])
            else:
                atr_14_daily[i] = (atr_14_daily[i-1] * 13 + tr[i]) / 14
    
    highest_high_14 = np.full(len(close_daily), np.nan)
    lowest_low_14 = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(14, len(close_daily)):
            highest_high_14[i] = np.max(high_daily[i-13:i+1])
            lowest_low_14[i] = np.min(low_daily[i-13:i+1])
    
    chop_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(14, len(close_daily)):
            if (not np.isnan(atr_14_daily[i]) and 
                not np.isnan(highest_high_14[i]) and 
                not np.isnan(lowest_low_14[i]) and
                highest_high_14[i] > lowest_low_14[i]):
                sum_atr = np.nansum(atr_14_daily[i-13:i+1])
                chop_daily[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high_14[i] - lowest_low_14[i]) + 1e-10)
            else:
                chop_daily[i] = 50  # neutral when no range
    
    # Align daily indicators to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    R2_aligned = align_htf_to_ltf(prices, df_daily, R2)
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    S2_aligned = align_htf_to_ltf(prices, df_daily, S2)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    chop_daily_aligned = align_htf_to_ltf(prices, df_daily, chop_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(chop_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 12h volume > 2x 20-day average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with volume spike in non-choppy market
            # Choppiness < 38.2 indicates trending market
            trending_market = chop_daily_aligned[i] < 38.2
            
            # Long when price breaks above R1 with volume spike
            long_condition = (
                close[i] > R1_aligned[i] and   # price above R1 pivot
                trending_market and            # trending market (not choppy)
                vol_spike                      # volume spike for momentum
            )
            
            # Short when price breaks below S1 with volume spike
            short_condition = (
                close[i] < S1_aligned[i] and   # price below S1 pivot
                trending_market and            # trending market (not choppy)
                vol_spike                      # volume spike for momentum
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R1 or market becomes choppy
            if close[i] < R1_aligned[i] or chop_daily_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S1 or market becomes choppy
            if close[i] > S1_aligned[i] or chop_daily_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals