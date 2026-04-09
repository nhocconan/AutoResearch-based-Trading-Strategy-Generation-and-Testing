#!/usr/bin/env python3
# 4h_donchian_breakout_12h_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with 12h volume confirmation and choppiness regime filter.
# Works in bull/bear: Donchian(20) captures breakouts; 12h volume > 1.5x average confirms institutional participation;
# Choppiness index (CHOP) > 61.8 avoids whipsaws in ranging markets. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for volume and chop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need sufficient data for volume MA and chop
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h volume confirmation: current volume > 1.5x 20-period average
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # 12h Choppiness Index (CHOP) - avoids ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest high - lowest low)))
    # We use a simplified version: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align with index
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    chop_denominator = np.log10(14) * (highest_high_12h - lowest_low_12h)
    chop_numerator = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    chop_12h = 100 * (chop_numerator / chop_denominator)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 4h Donchian Channel (20-period)
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(volume_ma_12h_aligned[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: avoid ranging markets (CHOP > 61.8)
        if chop_12h_aligned[i] > 61.8:
            # In ranging market, reduce position or stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR volume dries up
            if close[i] < lowest_low_4h[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR volume dries up
            if close[i] > highest_high_4h[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long: price breaks above Donchian upper band
                if close[i] > highest_high_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band
                elif close[i] < lowest_low_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals