#!/usr/bin/env python3
# 4h_donchian_volume_chop_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Works in bull/bear: Donchian captures breakouts, volume confirms institutional participation,
# choppiness index (CHOP>61.8) filters range markets to avoid false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_v2"
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
    
    # 1d HTF data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for CHOP
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for choppiness calculation
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])  # First TR is infinite
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d True Range sum and Price Change for Choppiness Index
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    pc = np.abs(close_1d[14:] - close_1d[:-14])  # 14-period price change
    pc_padded = np.concatenate([np.full(14, np.nan), pc])  # Pad to align with close_1d
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum / (atr * 14)) / log10(14)
    # Higher CHOP = more choppy (range), Lower CHOP = more trending
    chop_raw = 100 * np.log10(tr_sum / (atr_1d * 14)) / np.log10(14)
    chop_1d = np.concatenate([np.full(14, np.nan), chop_raw])  # Align with close_1d
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian(20) channels
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian Low OR chop becomes too low (trending exhaustion)
            if close[i] < dc_low[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian High OR chop becomes too low (trending exhaustion)
            if close[i] > dc_high[i] or chop_1d_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and chop > 61.8 (range regime)
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            chop_confirmed = chop_1d_aligned[i] > 61.8
            
            if volume_confirmed and chop_confirmed:
                # Long: price breaks above Donchian High
                if close[i] > dc_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian Low
                elif close[i] < dc_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals