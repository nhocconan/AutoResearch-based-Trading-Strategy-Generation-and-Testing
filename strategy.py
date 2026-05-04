#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and choppiness regime to avoid whipsaw in ranging markets. Designed for 20-50
# trades/year (~80-200 total over 4 years) to minimize fee drag. Volume spike
# ensures institutional participation, chop filter (CHOP > 61.8) enables mean reversion
# in ranges while breakouts work in trends (CHOP < 38.2). Works in bull/bear by
# adapting to regime: breakouts in trends, mean reversion at channel extremes in ranges.

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
    
    # Calculate 1d volume ratio (current / 20-period average) for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = np.where(vol_ma_20 > 0, volume_1d / vol_ma_20, 1.0)
    
    # Calculate 1d choppiness index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(TR over n) / (HHV(high,n) - LLV(low,n))) / log10(n)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hhvs = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvs = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = hhvs - llvs
    chop_1d = np.where(chop_denom > 0, 
                       100 * np.log10(atr_14 / chop_denom) / np.log10(14), 
                       50.0)  # Neutral when range=0
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    volume_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime-based logic
        if chop_1d_aligned[i] < 38.2:  # Trending regime - follow breakouts
            if position == 0:
                # Long breakout: price above upper Donchian + volume spike
                if (close[i] > highest_high[i] and 
                    volume_ratio_1d_aligned[i] > 2.0):
                    signals[i] = 0.30
                    position = 1
                # Short breakout: price below lower Donchian + volume spike
                elif (close[i] < lowest_low[i] and 
                      volume_ratio_1d_aligned[i] > 2.0):
                    signals[i] = -0.30
                    position = -1
            elif position == 1:
                # Exit long: price re-enters Donchian channel OR volume drops
                if (close[i] <= highest_high[i] and close[i] >= lowest_low[i]) or \
                   volume_ratio_1d_aligned[i] < 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Exit short: price re-enters Donchian channel OR volume drops
                if (close[i] <= highest_high[i] and close[i] >= lowest_low[i]) or \
                   volume_ratio_1d_aligned[i] < 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
                    
        elif chop_1d_aligned[i] > 61.8:  # Ranging regime - mean reversion at extremes
            if position == 0:
                # Long mean reversion: price at lower Donchian + volume spike
                if (close[i] <= lowest_low[i] * 1.001 and  # Allow tiny buffer
                    volume_ratio_1d_aligned[i] > 2.0):
                    signals[i] = 0.30
                    position = 1
                # Short mean reversion: price at upper Donchian + volume spike
                elif (close[i] >= highest_high[i] * 0.999 and  # Allow tiny buffer
                      volume_ratio_1d_aligned[i] > 2.0):
                    signals[i] = -0.30
                    position = -1
            elif position == 1:
                # Exit long: price reaches midpoint OR volume drops
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] >= midpoint or volume_ratio_1d_aligned[i] < 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Exit short: price reaches midpoint OR volume drops
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] <= midpoint or volume_ratio_1d_aligned[i] < 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
        else:  # Neutral regime (38.2 <= CHOP <= 61.8) - reduce activity
            if position != 0:
                signals[i] = 0.0
                position = 0
            # Stay flat in neutral regime to avoid whipsaw
    
    return signals