#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Works in bull markets via breakouts and bear markets via mean reversion in ranging conditions.
# Volume confirmation reduces false breakouts. Chop filter ensures trades only in favorable regimes.
# Discrete position sizing (0.0, ±0.25) minimizes fee churn. Target: 75-200 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
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
    
    # 1d HTF data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True range approximation for 1d: high-low (simplified for daily)
    tr_1d = high_1d - low_1d
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index: 100 * log10(sum(tr14) / (log10(14) * (hh14 - ll14))) / log10(14)
    # Avoid division by zero and log of zero/negative
    range_14 = hh_14 - ll_14
    range_14 = np.where(range_14 <= 0, 1e-10, range_14)
    log_atr = np.log10(np.maximum(atr_14, 1e-10))
    log_range = np.log10(range_14)
    log_n = np.log10(14)
    
    # Chop formula: 100 * log10(atr_sum / (log_n * range)) / log_n
    chop = 100 * (log_atr - (log_n + log_range)) / log_n
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral if undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging or weakly trending (chop between 30 and 70)
        chop_regime = (chop_aligned[i] > 30) & (chop_aligned[i] < 70)
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < lowest_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > highest_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian high
                if close[i] > highest_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low
                elif close[i] < lowest_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals