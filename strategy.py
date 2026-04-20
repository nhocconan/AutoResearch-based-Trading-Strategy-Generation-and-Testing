#!/usr/bin/env python3
# 6h_KAMA_Trend_With_1D_VolumeSpike_and_ChopFilter
# Hypothesis: 6h KAMA trend direction, filtered by 1d volume spike (vol > 1.5x MA20) and chop regime (CHOP(14) < 38.2 = trending).
# In choppy markets (CHOP > 61.8) we avoid trades to reduce whipsaw.
# Volume spike confirms institutional interest in breakout direction.
# KAMA adapts to market noise - faster in trends, slower in ranges.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_KAMA_Trend_With_1D_VolumeSpike_and_ChopFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter (volume > 1.5x 20-period MA)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate 1d Chopiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_period = 14
    atr = np.full_like(high_1d, np.nan)
    if len(high_1d) >= atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(high_1d)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Sum of TR over atr_period
    tr_sum = np.full_like(high_1d, np.nan)
    if len(high_1d) >= atr_period:
        tr_sum[atr_period-1] = np.nansum(tr[1:atr_period+1])
        for i in range(atr_period, len(high_1d)):
            tr_sum[i] = tr_sum[i-1] + tr[i] - tr[i-atr_period+1]
    
    # Chop calculation: 100 * log10(sum(TR) / (max(HH) - min(LL))) / log10(atr_period)
    chop = np.full_like(high_1d, 50.0)  # default neutral
    if len(high_1d) >= atr_period:
        # Highest high and lowest low over atr_period
        hh = np.full_like(high_1d, np.nan)
        ll = np.full_like(high_1d, np.nan)
        
        for i in range(atr_period-1, len(high_1d)):
            hh[i] = np.max(high_1d[i-atr_period+1:i+1])
            ll[i] = np.min(low_1d[i-atr_period+1:i+1])
        
        # Avoid division by zero
        range_hl = hh - ll
        valid = (range_hl > 0) & ~np.isnan(tr_sum) & ~np.isnan(hh) & ~np.isnan(ll)
        chop[valid] = 100 * np.log10(tr_sum[valid] / range_hl[valid]) / np.log10(atr_period)
    
    # Chop filter: trending when CHOP < 38.2
    chop_trending = chop < 38.2
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_trending.astype(float))
    
    # Calculate KAMA on 6h data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.concatenate([[np.nan], np.diff(close[:-1])]))  # |close[i] - close[i-1]|
    for i in range(1, len(change)):
        change[i] = np.abs(close[i] - close[i-1])
    
    # Sum of absolute changes over 10 periods
    abs_change_sum = np.full_like(close, np.nan)
    er_period = 10
    if len(close) >= er_period:
        abs_change_sum[er_period-1] = np.nansum(np.abs(np.diff(close[max(0, er_period-1):er_period]))) if er_period > 0 else 0
        for i in range(er_period, len(close)):
            abs_change_sum[i] = abs_change_sum[i-1] + np.abs(close[i] - close[i-1]) - np.abs(close[i-er_period] - close[i-er_period-1]) if i-er_period-1 >= 0 else abs_change_sum[i-1] + np.abs(close[i] - close[i-1])
    
    # Efficiency Ratio
    er = np.full_like(close, 0.0)
    price_change = np.abs(np.concatenate([[np.nan], np.diff(close[:-1])]))
    for i in range(1, len(price_change)):
        price_change[i] = np.abs(close[i] - close[i-1])
    
    valid_er = (~np.isnan(price_change)) & (~np.isnan(abs_change_sum)) & (abs_change_sum != 0)
    er[valid_er] = price_change[valid_er] / abs_change_sum[valid_er]
    
    # Smoothing constants
    fastest_sc = 2 / (2 + 1)  # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    if len(close) > 0:
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 20)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: KAMA trend + volume spike + chop filter (trending market)
        if position == 0:
            kama_up = close[i] > kama[i]
            kama_down = close[i] < kama[i]
            vol_spike_now = vol_spike_aligned[i] > 0.5  # boolean as float
            chop_filter_now = chop_filter_aligned[i] > 0.5  # trending market
            
            # Long: price above KAMA + volume spike + trending market
            if kama_up and vol_spike_now and chop_filter_now:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + volume spike + trending market
            elif kama_down and vol_spike_now and chop_filter_now:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or conditions fail
            if close[i] <= kama[i] or vol_spike_aligned[i] <= 0.5 or chop_filter_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or conditions fail
            if close[i] >= kama[i] or vol_spike_aligned[i] <= 0.5 or chop_filter_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals