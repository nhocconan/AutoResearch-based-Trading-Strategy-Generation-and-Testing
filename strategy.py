#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation.
# Uses 1d Choppiness Index to detect ranging (CHOP > 61.8) and trending (CHOP < 38.2) markets.
# In trending regimes, trades Donchian breakouts with volume confirmation.
# In ranging regimes, fades Donchian breakouts (mean reversion at channel boundaries).
# Designed to work in both bull and bear markets by adapting to market regime.
# Target: 12-30 trades/year per symbol to avoid excessive fee drag.
name = "12h_Choppiness_Donchian_Breakout_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.full_like(tr, np.nan)
    for i in range(27, len(tr)):  # Need 14+13=27 periods for first valid value
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr_14[i] = np.nansum(atr_14[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = np.full_like(high_1d, np.nan)
    lowest_low_14 = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        highest_high_14[i] = np.nanmax(high_1d[i-13:i+1])
        lowest_low_14[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    for i in range(27, len(close_1d)):
        if (not np.isnan(sum_atr_14[i]) and 
            not np.isnan(highest_high_14[i]) and 
            not np.isnan(lowest_low_14[i]) and
            highest_high_14[i] != lowest_low_14[i]):
            chop[i] = 100 * np.log10(sum_atr_14[i] / (highest_high_14[i] - lowest_low_14[i])) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 12h data
    highest_high_20 = np.full_like(high, np.nan)
    lowest_low_20 = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        highest_high_20[i] = np.nanmax(high[i-19:i+1])
        lowest_low_20[i] = np.nanmin(low[i-19:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ema_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ema_20[i] = np.nanmean(volume[i-20:i])
    vol_ratio = np.where(vol_ema_20 > 0, volume / vol_ema_20, 0)
    vol_spike = vol_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        trending = chop_aligned[i] < 38.2
        ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            if trending:
                # In trending regime: breakout continuation
                long_condition = (close[i] > highest_high_20[i]) and vol_spike[i]
                short_condition = (close[i] < lowest_low_20[i]) and vol_spike[i]
            elif ranging:
                # In ranging regime: fade at channel boundaries (mean reversion)
                long_condition = (close[i] <= lowest_low_20[i]) and vol_spike[i]
                short_condition = (close[i] >= highest_high_20[i]) and vol_spike[i]
            else:
                # Choppy middle region: no trade
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions
            if trending:
                # Exit long when price crosses below Donchian low or trend ends
                if (close[i] < lowest_low_20[i]) or (not trending):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif ranging:
                # Exit long when price reaches Donchian high or range ends
                if (close[i] >= highest_high_20[i]) or (not ranging):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit conditions
            if trending:
                # Exit short when price crosses above Donchian high or trend ends
                if (close[i] > highest_high_20[i]) or (not trending):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif ranging:
                # Exit short when price reaches Donchian low or range ends
                if (close[i] <= lowest_low_20[i]) or (not ranging):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals