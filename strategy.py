#!/usr/bin/env python3
# 12h_donchian_breakout_1d_volume_chop_v1
# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Works in bull/bear: Donchian captures breakouts, volume confirms validity, chop filter avoids whipsaws in ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_volume_chop_v1"
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
    
    # 1d HTF data for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need 20-period for volume MA and chop
        return np.zeros(n)
    
    # 1d volume and chop filter
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d volume MA (20-period)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d ATR (14-period) for choppiness calculation
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr1 = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr2 = np.concatenate([[np.abs(high_1d[0] - close_1d[0])], tr2])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr3 = np.concatenate([[np.abs(low_1d[0] - close_1d[0])], tr3])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d True Range sum (14-period) for denominator
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # 1d High-Low range (14-period) for numerator
    hl_range = high_1d - low_1d
    hl_sum_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / hl_sum_14) / log10(14)
    # Avoid division by zero and log of zero
    chop_raw = np.where(
        (hl_sum_14 > 0) & (tr_sum_14 > 0),
        100 * np.log10(tr_sum_14 / hl_sum_14) / np.log10(14),
        50.0  # neutral value when invalid
    )
    
    # Align 1d indicators to 12h timeframe (completed 1d bar only)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR chop too high (range-bound)
            if close[i] < lowest_low[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR chop too high (range-bound)
            if close[i] > highest_high[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and chop filter (trending market)
            volume_confirmed = volume[i] > 1.5 * vol_ma_aligned[i]
            chop_filter = chop_aligned[i] < 61.8  # trending market
            
            if volume_confirmed and chop_filter:
                # Long: price breaks above Donchian upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals