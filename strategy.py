#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Uses Donchian channel breakouts as price structure signals, confirmed by volume spikes
# and filtered by choppiness index to avoid ranging markets. Designed for 20-50 trades/year
# (~80-200 total over 4 years) to minimize fee drag. Works in both bull/bear markets by
# adapting to chop regime: trend follow when CHOP < 38.2, mean revert when CHOP > 61.8.

name = "4h_Donchian20_VolumeChop_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for choppiness denominator
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d True Range sum and ATR sum for choppiness
    tr_sum_1d = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr_sum_1d = atr_1d * 14
    
    # Calculate 1d Choppiness Index: CHOP = 100 * log10(TR_sum / (ATR * 14)) / log10(14)
    chop_1d = 100 * np.log10(tr_sum_1d / atr_sum_1d) / np.log10(14)
    
    # Align choppiness to 4h timeframe (wait for completed 1d bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume SMA(20) for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_sma[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop = chop_1d_aligned[i]
        vol_spike = volume[i] > 2.0 * vol_sma[i]  # Volume > 2x average
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            if chop < 38.2:  # Trending regime - follow Donchian breakouts
                # Long: price breaks above Donchian high with volume confirmation
                if close[i] > donch_high[i] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with volume confirmation
                elif close[i] < donch_low[i] and vol_spike:
                    signals[i] = -0.25
                    position = -1
            elif chop > 61.8:  # Ranging regime - mean revert at Donchian boundaries
                # Long: price bounces off Donchian low with volume confirmation
                if close[i] < donch_low[i] * 1.005 and vol_spike and close[i-1] >= donch_low[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: price bounces off Donchian high with volume confirmation
                elif close[i] > donch_high[i] * 0.995 and vol_spike and close[i-1] <= donch_high[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume drops
            if donch_low[i] <= close[i] <= donch_high[i] or volume[i] < 0.5 * vol_sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume drops
            if donch_low[i] <= close[i] <= donch_high[i] or volume[i] < 0.5 * vol_sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals