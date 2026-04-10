#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Long when price breaks above 12h Donchian upper band AND 1d volume > 1.5x 20-period volume SMA AND 1d chop > 61.8 (range)
# - Short when price breaks below 12h Donchian lower band AND 1d volume > 1.5x 20-period volume SMA AND 1d chop > 61.8 (range)
# - Exit: price retreats to 12h Donchian midpoint or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Donchian channels from 12h for structure, 1d for volume and regime confirmation

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d chopiness index (Ehler's Chop)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Chopiness index: 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop = np.where(
        (range_14 > 0) & (~np.isnan(sum_tr_14)) & (~np.isnan(atr_14)),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        np.nan
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align 1d volume SMA to 12h timeframe
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA (using 1d SMA as proxy)
        vol_confirm = volume[i] > 1.5 * volume_sma_20_12h_aligned[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        regime_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above previous upper band
        breakout_down = close[i] < low_min_20[i-1]  # Break below previous lower band
        
        # Exit conditions: price retreats to midpoint or loss of volume/regime
        exit_long = close[i] < donchian_mid[i] or not (vol_confirm and regime_filter)
        exit_short = close[i] > donchian_mid[i] or not (vol_confirm and regime_filter)
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals