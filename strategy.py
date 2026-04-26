#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime
Hypothesis: Donchian(20) breakout with volume spike (top 20%) and choppiness regime (CHOP < 38.2 = trending) to capture strong trends. Uses discrete position sizing (0.30) to limit trades and reduce fee drag. Works in both bull and bear markets by following established trends with volume confirmation and avoiding choppy, ranging markets where breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for HTF trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: volume > 80th percentile of 20-period lookback (volume spike)
    vol_series = pd.Series(volume)
    vol_percentile_80 = vol_series.rolling(window=20, min_periods=20).quantile(0.80).values
    volume_spike = volume > vol_percentile_80
    
    # Choppiness regime: CHOP < 38.2 = trending market (good for breakouts)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log(highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_regime = chop < 38.2  # trending market
    
    # Fixed position size to control trade frequency
    fixed_size = 0.30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and Donchian, 20 for volume percentile, 14 for ATR/CHOP)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_percentile_80[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        ema_1d_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_reg = chop_regime[i]
        size = fixed_size
        
        # Entry conditions: Donchian breakout with volume spike AND trending regime
        long_entry = (close_val > upper_channel) and vol_spike and chop_reg and (close_val > ema_1d_trend)
        short_entry = (close_val < lower_channel) and vol_spike and chop_reg and (close_val < ema_1d_trend)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian lower channel break (trend reversal)
            if close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian upper channel break (trend reversal)
            if close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0