#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and 1w choppiness regime filter
# - Enter long when price breaks above 20-period Donchian upper band AND 1d volume > 1.5x 20-period volume SMA AND 1w chop < 61.8 (trending regime)
# - Enter short when price breaks below 20-period Donchian lower band AND 1d volume > 1.5x 20-period volume SMA AND 1w chop < 61.8 (trending regime)
# - Exit: price reverses to opposite Donchian band (upper for shorts, lower for longs)
# - Donchian breakout captures strong momentum moves
# - Volume confirmation ensures institutional participation
# - 1w chop filter avoids false breakouts in ranging markets
# - Target: 25-40 trades/year to minimize fee drag while capturing high-probability trends

name = "4h_1d_1w_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for choppiness regime filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 20-period Donchian channels for 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute True Range for 1w choppiness calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.abs(high_1w[0] - low_1w[0])  # First period
    tr2[0] = 0  # No previous close for first period
    tr3[0] = 0  # No previous close for first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 14-period ATR and sum of ranges for choppiness
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_hh_14 - min_ll_14
    
    # Choppiness Index: 100 * log10(sum TR / range) / log10(14)
    # Avoid division by zero and log of zero
    chop_raw = np.where((range_14 > 0) & (sum_tr_14 > 0), 
                        100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 
                        50)  # Neutral value when undefined
    
    # Align indicators to 4h timeframe
    volume_1d_current = df_1d['volume'].values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: 1w chop < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        
        # Donchian breakout signals
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # Exit signals: price reverses to opposite Donchian band
        long_exit = close[i] < lowest_low_20[i]  # Exit long when price breaks lower band
        short_exit = close[i] > highest_high_20[i]  # Exit short when price breaks upper band
        
        # Trading logic
        if long_breakout and vol_confirm and trending_regime:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif short_breakout and vol_confirm and trending_regime:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Check for exits
            if position == 1 and long_exit:
                position = 0
                signals[i] = 0.0
            elif position == -1 and short_exit:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals