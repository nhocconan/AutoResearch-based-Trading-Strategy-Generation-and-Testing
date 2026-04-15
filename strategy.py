#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) Breakout + 1d Volume Spike + 1d Choppiness Filter
# Uses 12h Donchian breakout for directional bias, confirmed by 1d volume spike (>1.5x 20-day median)
# and filtered by 1d Choppiness Index (CHOP > 61.8 = range, avoid breakouts in chop).
# Designed to capture strong trending moves while avoiding false breakouts in sideways markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day Volume Spike: > 1.5x 20-day median volume
    vol_median_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median()
    vol_threshold_1d = 1.5 * vol_median_1d
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, volume_1d > vol_threshold_1d)
    
    # 1-day Choppiness Index (CHOP) - avoids breakouts in ranging markets
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    chop_filter = align_htf_to_ltf(prices, df_1d, chop < 61.8)  # Only allow breakouts when NOT choppy (trending)
    
    # 12-hour Donchian Channel (20-period)
    # Since we're on 12h timeframe, we need 20*12h = 10 days of data
    # But we'll use the 12h data directly from prices
    highest_high_12h = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low_12h = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Breakout signals
    breakout_up = close > highest_high_12h.shift(1)
    breakout_down = low < lowest_low_12h.shift(1)
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup for Donchian
        # Skip if any required data is NaN
        if (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or
            np.isnan(vol_spike_1d[i]) or np.isnan(chop_filter[i])):
            continue
        
        # Long: 12h breakout up + volume spike + not choppy
        if breakout_up[i] and vol_spike_1d[i] and chop_filter[i]:
            signals[i] = 0.25
        
        # Short: 12h breakout down + volume spike + not choppy
        elif breakout_down[i] and vol_spike_1d[i] and chop_filter[i]:
            signals[i] = -0.25
        
        # Exit: opposite breakout or loss of volume spike or choppy conditions
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (breakout_down[i] or not vol_spike_1d[i] or not chop_filter[i])) or
               (signals[i-1] == -0.25 and (breakout_up[i] or not vol_spike_1d[i] or not chop_filter[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian20_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0