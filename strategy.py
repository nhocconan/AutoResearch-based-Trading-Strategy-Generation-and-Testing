#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high in low-chop regime (CHOP < 40) with volume spike
# - Short when price breaks below Donchian(20) low in low-chop regime (CHOP < 40) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Chop regime filter avoids whipsaws in ranging markets
# - Volume confirmation ensures breakout validity
# - Works in both bull and bear markets by filtering for trending conditions via chop

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR for Donchian calculation
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1d = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # Pre-compute 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(period))
    # Simplified: CHOP = 100 * log10( sum(ATR14 over period) / (HH - LL) ) / log10(period)
    chop_period = 14
    sum_atr_14 = pd.Series(true_ranges).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(df_1d['high'].values).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=chop_period, min_periods=chop_period).min().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop_raw = np.log10(sum_atr_14 / range_hl) / np.log10(chop_period) * 100
    chop_values = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50.0, chop_raw)  # Default to neutral
    
    # Align all HTF indicators to LTF
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop_multiplier = 2.5
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # ATR-based stoploss
            if prices['close'].iloc[i] < entry_price - atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # ATR-based stoploss
            if prices['close'].iloc[i] > entry_price + atr_stop_multiplier * atr_14_1d_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long signal: price breaks above Donchian high in low-chop regime with volume spike
            if (prices['high'].iloc[i] > donchian_high_20_aligned[i] and 
                chop_aligned[i] < 40.0 and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low in low-chop regime with volume spike
            elif (prices['low'].iloc[i] < donchian_low_20_aligned[i] and 
                  chop_aligned[i] < 40.0 and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals