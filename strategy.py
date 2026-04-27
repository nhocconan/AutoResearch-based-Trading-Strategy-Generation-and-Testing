#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_Regime_Filter
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation (>2.0x 20-period average), and choppiness regime filter (CHOP < 38.2 for trending markets). 
Enter long when price breaks above Donchian upper band in 1d uptrend with volume spike and trending regime. 
Enter short when price breaks below Donchian lower band in 1d downtrend with volume spike and trending regime. 
Exit on opposite Donchian band touch or 1d trend reversal. 
Designed for moderate trade frequency (~30-60/year) with discrete position sizing (0.25) to minimize fee drag while capturing strong trends. 
Works in bull markets via long breakouts and in bear markets via short breakdowns, with regime filter avoiding sideways whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d choppiness regime: CHOP(14) < 38.2 = trending (use for filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.absolute(high_1d - np.roll(close_1d, 1)),
                    np.absolute(low_1d - np.roll(close_1d, 1)))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = sum_tr_14 + 1e-10
    chop = 100 * np.log10(chop_denom / (highest_high_14 - lowest_low_14 + 1e-10)) / np.log10(14)
    chop_regime = chop < 38.2  # trending market
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Donchian(20) channels on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA50 (50), Donchian (20), volume avg (20), chop (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(chop_regime_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_donch = highest_high_20[i]
        lower_donch = lowest_low_20[i]
        ema_val = ema_50_aligned[i]
        chop_regime_val = bool(chop_regime_aligned[i])
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1d EMA50 trend filter, volume spike, and trending regime
            # Long: price breaks above upper Donchian AND above EMA50 (1d uptrend) AND volume spike AND trending regime
            long_condition = (close_val > upper_donch) and (close_val > ema_val) and vol_conf and chop_regime_val
            # Short: price breaks below lower Donchian AND below EMA50 (1d downtrend) AND volume spike AND trending regime
            short_condition = (close_val < lower_donch) and (close_val < ema_val) and vol_conf and chop_regime_val
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian (opposite band) OR 1d EMA50 turns bearish (price below EMA) OR regime turns choppy
            if (close_val < lower_donch) or (close_val < ema_val) or not chop_regime_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches upper Donchian (opposite band) OR 1d EMA50 turns bullish (price above EMA) OR regime turns choppy
            if (close_val > upper_donch) or (close_val > ema_val) or not chop_regime_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_Regime_Filter"
timeframe = "4h"
leverage = 1.0