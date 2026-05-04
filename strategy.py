#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and filtered by choppiness regime (CHOP > 61.8 = range, avoid breakouts in chop).
# Designed for 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns.
# Volume spike confirms institutional interest, chop filter avoids false breakouts in ranging markets.

name = "4h_Donchian20_1dVolumeSpike_ChopRegime"
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
    
    # Get 1d data for volume spike and choppiness - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(TR)/ (HHV - LLV)) / log10(14)
    # Where TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hhvs = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvs = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_14 * 14 / (hhvs - llvs + 1e-10)) / np.log10(14)
    chop_regime_range = chop_raw > 61.8  # choppy/range market
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_range.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous period high
        breakout_down = close[i] < lowest_low[i-1]   # break below previous period low
        
        if position == 0:
            # Long: bullish breakout + volume spike + NOT in choppy regime
            if (breakout_up and 
                volume_spike_aligned[i] > 0.5 and 
                chop_regime_aligned[i] < 0.5):  # not choppy (trending)
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown + volume spike + NOT in choppy regime
            elif (breakout_down and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_regime_aligned[i] < 0.5):  # not choppy (trending)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR opposite breakdown
            if (close[i] <= highest_high[i] and close[i] >= lowest_low[i]) or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR opposite breakout
            if (close[i] <= highest_high[i] and close[i] >= lowest_low[i]) or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals