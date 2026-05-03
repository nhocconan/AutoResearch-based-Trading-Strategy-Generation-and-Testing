#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d Volume Spike filter and Chop regime.
# Long: Close breaks above Donchian upper AND 1d volume > 2.0x 20-period MA AND Chop(14) > 61.8 (range) for mean reversion
# Short: Close breaks below Donchian lower AND 1d volume > 2.0x 20-period MA AND Chop(14) > 61.8 (range) for mean reversion
# Exit: Opposite Donchian breakout or Chop < 38.2 (trend) or volume drops below 1.5x MA.
# Uses 1d volume spike to confirm institutional interest, Chop regime to ensure mean-reversion logic works in ranging markets.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dVolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and Chop regime
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA (20-period)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ma_20)
    
    # Calculate 1d Chop index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR (14-period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA (for entry/exit timing)
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.5 * vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        chop_val = chop_aligned[i]
        vol_spike_1d = volume_spike_aligned[i]
        vol_spike_4h = volume_spike_4h[i]
        
        # Determine Chop regime
        is_range = chop_val > 61.8  # Chop > 61.8 = ranging (mean revert)
        is_trend = chop_val < 38.2  # Chop < 38.2 = trending
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND 1d volume spike AND ranging (mean reversion long)
            if close_val > donchian_upper[i] and vol_spike_1d and is_range:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower AND 1d volume spike AND ranging (mean reversion short)
            elif close_val < donchian_lower[i] and vol_spike_1d and is_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian lower OR trend develops (Chop < 38.2) OR 4h volume drops
            if close_val < donchian_lower[i] or is_trend or not vol_spike_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian upper OR trend develops (Chop < 38.2) OR 4h volume drops
            if close_val > donchian_upper[i] or is_trend or not vol_spike_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals