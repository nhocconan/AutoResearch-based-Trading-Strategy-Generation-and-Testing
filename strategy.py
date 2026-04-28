#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width Regime + 1d Donchian(20) Breakout with Volume Spike
# Uses BB Width percentile to detect regime: low BB Width (<20th percentile) = squeeze (range),
# high BB Width (>80th percentile) = expansion (trend). Only trade breakouts in expansion regime.
# 1d Donchian(20) breakout provides directional bias. Volume spike confirms institutional participation.
# Works in both bull and bear markets by only trading high-probability breakouts during volatile regimes.
# Target: 12-30 trades/year via tight regime + breakout + volume confluence.

name = "6h_BBWidth_Regime_1dDonchian20_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian breakout calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    dh_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 6h timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1d, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1d, dl_20)
    
    # Calculate 6h Bollinger Bands (20, 2) for regime detection
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2 * dev
    lower_bb = basis - 2 * dev
    bb_width = (upper_bb - lower_bb) / basis  # Normalized width
    
    # Calculate BB Width percentile rank (lookback=50) to detect regime
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: >2.0x 20-bar average volume (institutional participation)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for BB Width percentile and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        bb_percentile = bb_width_percentile[i]
        price = close[i]
        
        # Regime filter: only trade in expansion regime (BB Width > 80th percentile)
        in_expansion = bb_percentile > 80
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when: price breaks above 1d Donchian high AND expansion regime AND volume spike
            if price > dh_20_aligned[i] and in_expansion and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when: price breaks below 1d Donchian low AND expansion regime AND volume spike
            elif price < dl_20_aligned[i] and in_expansion and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price breaks below Donchian low or regime changes to squeeze
            if price < dl_20_aligned[i] or not in_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price breaks above Donchian high or regime changes to squeeze
            if price > dh_20_aligned[i] or not in_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals