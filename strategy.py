#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND chop > 61.8.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 20-50 trades/year.
# Donchian breakout captures momentum, volume spike confirms institutional interest, chop filter avoids whipsaws in strong trends.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) by requiring both conditions.

name = "4h_Donchian20_1dVolumeSpike_ChopFilter_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike: current volume > 1.5x 20-period average
    vol_20ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.5 * vol_20ma)
    
    # Calculate 1d chopiness index: CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(N)
    # Simplified: CHOP > 61.8 = range, CHOP < 38.2 = trend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr_1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # sum of ATR(1) over period
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    chop = 100 * np.log10(atr_1 / range_14) / np.log10(14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid division by zero
    
    # Align 1d indicators to 4h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
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
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_volume_spike = volume_spike_aligned[i] > 0.5  # boolean
        curr_chop = chop_aligned[i]
        
        # Chop regime: only trade in range markets (CHOP > 61.8)
        in_range = curr_chop > 61.8
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND volume spike AND in range
            if (curr_close > curr_high_20 and 
                curr_volume_spike and 
                in_range):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND in range
            elif (curr_close < curr_low_20 and 
                  curr_volume_spike and 
                  in_range):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR loss of volume spike OR chop < 38.2 (trend)
            if (curr_close < curr_low_20 or 
                not curr_volume_spike or 
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR loss of volume spike OR chop < 38.2 (trend)
            if (curr_close > curr_high_20 or 
                not curr_volume_spike or 
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals