#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg volume AND chop > 61.8 (range regime).
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg volume AND chop > 61.8.
# Uses discrete sizing 0.25. Target: 20-50 trades/year.
# Donchian provides structural breakouts, volume confirms conviction, chop filter avoids whipsaws in strong trends.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) by focusing on high-conviction moves in ranging markets.

name = "4h_Donchian20_VolumeConfirm_ChopRegime_v1"
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
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) sum
    atr_period = 14
    atr_sum = np.full_like(close_1d, np.nan)
    for i in range(atr_period, len(close_1d)):
        atr_sum[i] = np.nansum(tr[i-atr_period+1:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.full_like(close_1d, np.nan)
    ll = np.full_like(close_1d, np.nan)
    for i in range(atr_period-1, len(close_1d)):
        hh[i] = np.nanmax(high_1d[i-atr_period+1:i+1])
        ll[i] = np.nanmin(low_1d[i-atr_period+1:i+1])
    
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(atr_period)
    chop = np.full_like(close_1d, np.nan)
    for i in range(atr_period, len(close_1d)):
        if hh[i] > ll[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(atr_period)
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    for i in range(lookback-1, len(close)):
        highest_high[i] = np.nanmax(high[i-lookback+1:i+1])
        lowest_low[i] = np.nanmin(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-bar average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    vol_ratio = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for Donchian and volume
    
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_ratio = vol_ratio[i]
        curr_chop = chop_aligned[i]
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        in_range = curr_chop > 61.8
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND volume confirmation AND in ranging market
            if (curr_close > highest_high[i] and 
                curr_vol_ratio > 1.5 and 
                in_range):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmation AND in ranging market
            elif (curr_close < lowest_low[i] and 
                  curr_vol_ratio > 1.5 and 
                  in_range):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR loss of volume conviction OR exits ranging market
            if (curr_close < lowest_low[i] or 
                curr_vol_ratio < 1.2 or 
                not in_range):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR loss of volume conviction OR exits ranging market
            if (curr_close > highest_high[i] or 
                curr_vol_ratio < 1.2 or 
                not in_range):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals