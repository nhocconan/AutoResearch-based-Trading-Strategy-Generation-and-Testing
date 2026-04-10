#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1h chop regime filter
# - Long when price breaks above 4h Donchian upper band (20-period high) AND 1d volume > 1.5x 20-period volume SMA AND 1h chop > 61.8 (range regime)
# - Short when price breaks below 4h Donchian lower band (20-period low) AND 1d volume > 1.5x 20-period volume SMA AND 1h chop > 61.8 (range regime)
# - Exit: price retreats to midpoint of Donchian channel OR volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 15-40 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian structure from 4h, volume confirmation from 1d, regime filter from 1h for mean reversion in choppy markets

name = "4h_1d_1h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1h = get_htf_data(prices, '1h')
    if len(df_1d) < 30 or len(df_1h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1h chop index regime filter (Ehler's Chop: 100 * log10(sum(atr(14)) / log10(highest_high - lowest_low)) / log10(14))
    # Simplified: chop > 61.8 = ranging market (good for mean reversion), chop < 38.2 = trending
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate True Range for 1h
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr1[0] = high_1h[0] - low_1h[0]  # first bar
    tr2[0] = np.abs(high_1h[0] - close_1h[0])  # approximate
    tr3[0] = np.abs(low_1h[0] - close_1h[0])   # approximate
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods for chop denominator
    hh_14 = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr(14)) / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop_denominator = np.where(range_14 > 0, range_14, 1e-10)
    chop_ratio = sum_atr_14 / chop_denominator
    chop_ratio = np.where(chop_ratio > 0, chop_ratio, 1e-10)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align HTF data to 4h timeframe
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1h, chop)
    
    for i in range(40, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        # Need to get 1d volume at the aligned index
        vol_idx = i // 6  # Approximate: 4h bars per 1d (6 * 4h = 24h)
        if vol_idx < len(volume_1d):
            vol_confirm = volume_1d[vol_idx] > 1.5 * volume_sma_20_1d[vol_idx]
        else:
            vol_confirm = False
        
        # Chop regime filter: chop > 61.8 = ranging market (favorable for mean reversion breakouts)
        chop_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > high_max_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_min_20[i-1]  # Break below previous period's low
        
        # Exit conditions: price retreats to midpoint OR loss of volume/chop confirmation
        exit_long = close[i] < donchian_mid[i] or not (vol_confirm and chop_filter)
        exit_short = close[i] > donchian_mid[i] or not (vol_confirm and chop_filter)
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and chop_filter:
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