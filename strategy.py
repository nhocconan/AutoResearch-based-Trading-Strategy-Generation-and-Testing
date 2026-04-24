#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels (H3/L3) and volume confirmation.
- Camarilla levels: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low) from prior 1d candle.
- Regime: Choppiness Index(14) on 1d < 38.2 = trending (favor breakouts), > 61.8 = choppy (avoid).
- Entry: Long when price > H3 AND trending regime AND volume > 1.5 * 20-period average volume.
         Short when price < L3 AND trending regime AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in trending regimes, avoiding whipsaws in chop.
- Uses 1d HTF data called ONCE before loop with proper alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla H3/L3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume average and choppiness
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Prior day's Camarilla levels (H3, L3)
    # H3 = prior_close + 1.1 * (prior_high - prior_low)
    # L3 = prior_close - 1.1 * (prior_high - prior_low)
    prior_close = close_1d[:-1]  # t-1
    prior_high = high_1d[:-1]    # t-1
    prior_low = low_1d[:-1]      # t-1
    prior_range = prior_high - prior_low
    
    h3_1d = prior_close + 1.1 * prior_range
    l3_1d = prior_close - 1.1 * prior_range
    
    # Align Camarilla levels to 12h timeframe (1 completed 1d bar delay)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d Choppiness Index(14) for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(14) - sum of TR over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) and Min(low) over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(atr14 / (max_high_14 - min_low_14)) / log10(14)
    # Chop > 61.8 = ranging/choppy, Chop < 38.2 = trending
    ratio = atr14 / (max_high_14 - min_low_14 + 1e-10)  # Avoid division by zero
    chop = 100 * np.log10(ratio) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade breakouts in trending markets (Chop < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla breakout
        if position != 0:
            # Exit long: price < H3
            if position == 1:
                if curr_close < h3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3
            elif position == -1:
                if curr_close > l3_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > H3 AND trending regime AND volume confirmation
            long_condition = (curr_close > h3_1d_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < L3 AND trending regime AND volume confirmation
            short_condition = (curr_close < l3_1d_aligned[i] and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dVolSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0