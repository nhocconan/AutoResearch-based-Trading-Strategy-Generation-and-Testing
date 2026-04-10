#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above 12h Donchian upper (20) AND 1d volume > 1.5x 20-period volume SMA AND chop > 61.8 (range)
# - Short when price breaks below 12h Donchian lower (20) AND 1d volume > 1.5x 20-period volume SMA AND chop > 61.8 (range)
# - Exit: price retreats to 12h Donchian middle (median of upper/lower) OR chop < 38.2 (trend)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Donchian channels from 12h timeframe for structure, 1d for volume and regime confirmation

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    donchian_upper = high_max_20
    donchian_lower = low_min_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d chop regime filter (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # ATR(14) sum
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high - lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    hh_ll_14 = hh_14 - ll_14
    
    # Chop = 100 * log10(atr_sum / (hh_ll)) / log10(14)
    chop = 100 * np.log10(atr_14 / hh_ll_14) / np.log10(14)
    chop = np.where(hh_ll_14 == 0, 100, chop)  # Avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = False
        idx_1d = i // 2  # Approximate 12h to 1d mapping (2x 12h bars per day)
        if idx_1d < len(volume_1d) and not np.isnan(volume_sma_20_1d_aligned[i]):
            vol_confirm = volume_1d[idx_1d] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Chop regime: range market (chop > 61.8)
        chop_range = chop_aligned[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Exit conditions: price retreats to middle OR chop < 38.2 (trend)
        exit_long = close[i] < donchian_middle[i] or chop_aligned[i] < 38.2
        exit_short = close[i] > donchian_middle[i] or chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_range:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and chop_range:
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