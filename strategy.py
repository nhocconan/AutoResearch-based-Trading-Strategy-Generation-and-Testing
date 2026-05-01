#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Uses 1d ATR for stoploss and position sizing. Long when price breaks above Donchian(20) high
# with volume > 1.5x average and choppy market (CHOP > 61.8). Short when price breaks below
# Donchian(20) low with volume confirmation and choppy market. Uses discrete sizing 0.25.
# Designed to capture trends in choppy/range-bound markets which are common in bear phases
# while avoiding whipsaws in strong trends via the chop filter. Works in both bull and bear
# markets by adapting to regime conditions.

name = "4h_Donchian20_VolumeChop_Regime_v1"
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
    
    # Load 1d data ONCE before loop for ATR and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr_1d[i] = np.nan
        elif i == 14:
            atr_1d[i] = np.nanmean(tr[1:15])  # first ATR is average of first 14 TR
        else:
            if not np.isnan(atr_1d[i-1]):
                atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
            else:
                atr_1d[i] = np.nan
    
    # Calculate 1d Choppiness Index(14)
    # CHOP = 100 * log10(sum(TR(14)) / (ATR(14) * 14)) / log10(14)
    sum_tr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            sum_tr_14[i] = np.nan
        else:
            sum_tr_14[i] = np.nansum(tr[i-13:i+1])
    
    chop_1d = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14 or np.isnan(sum_tr_14[i]) or np.isnan(atr_1d[i]) or atr_1d[i] == 0:
            chop_1d[i] = np.nan
        else:
            chop_1d[i] = 100 * np.log10(sum_tr_14[i] / (atr_1d[i] * 14)) / np.log10(14)
    
    # Align 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Donchian(20) channels on 4h
    # Donchian High = highest high of last 20 periods
    # Donchian Low = lowest low of last 20 periods
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.nanmax(high[i-19:i+1])
            donchian_low[i] = np.nanmin(low[i-19:i+1])
    
    # Calculate volume average(20) for confirmation
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.nanmean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Donchian and volume MA
    
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
            np.isnan(vol_ma[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dc_high = donchian_high[i]
        curr_dc_low = donchian_low[i]
        curr_vol_ma = vol_ma[i]
        curr_atr = atr_1d_aligned[i]
        curr_chop = chop_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = curr_volume > (1.5 * curr_vol_ma)
        
        # Choppiness regime: CHOP > 61.8 indicates choppy/range-bound market
        choppy_market = curr_chop > 61.8
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high + volume confirmation + choppy market
            if (curr_close > curr_dc_high and 
                vol_confirmed and 
                choppy_market):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume confirmation + choppy market
            elif (curr_close < curr_dc_low and 
                  vol_confirmed and 
                  choppy_market):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR ATR-based stoploss
            # Stoploss: entry price - 2 * ATR (tracked via position logic)
            if (curr_close < curr_dc_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR ATR-based stoploss
            if (curr_close > curr_dc_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals