#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter.
# Uses 12h Donchian channel for structure breakout, confirmed by 1d volume spike (>2x 20-period average)
# and 1d choppiness index (CHOP > 61.8 for range, < 38.2 for trend) to avoid false breakouts in chop.
# Long when: price breaks above Donchian(20) high AND 1d volume spike AND 1d CHOP < 38.2 (trending)
# Short when: price breaks below Donchian(20) low AND 1d volume spike AND 1d CHOP < 38.2 (trending)
# Uses discrete sizing 0.25. Target: 12-37 trades/year.
# Donchian breakouts capture momentum; volume confirms conviction; CHOP filter avoids whipsaws in ranging markets.
# Works in bull (breakouts continuation) and bear (breakdowns continuation) by aligning with 1d structure.

name = "12h_Donchian20_1dVolumeSpike_CHOPTrendFilter_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike: volume > 2.0 * 20-period SMA
    vol_sma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_sma_20)
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # where sum is over 14 periods, n=14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR(14) over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    mask = (sum_atr_14 > 0) & (highest_high_14 > lowest_low_14) & (~np.isnan(sum_atr_14))
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / (np.log10(14) * (highest_high_14[mask] - lowest_low_14[mask])))
    
    # Align 1d indicators to 12h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian(20) channels
    # Donchian high = max(high, 20)
    # Donchian low = min(low, 20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and 1d indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_volume_spike = volume_spike_aligned[i] > 0.5  # boolean
        curr_chop = chop_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND volume spike AND trending regime (CHOP < 38.2)
            if (curr_close > curr_donchian_high and 
                curr_volume_spike and 
                curr_chop < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND trending regime (CHOP < 38.2)
            elif (curr_close < curr_donchian_low and 
                  curr_volume_spike and 
                  curr_chop < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR loss of volume momentum OR choppy regime (CHOP > 61.8)
            if (curr_close < curr_donchian_low or 
                not curr_volume_spike or 
                curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR loss of volume momentum OR choppy regime (CHOP > 61.8)
            if (curr_close > curr_donchian_high or 
                not curr_volume_spike or 
                curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals