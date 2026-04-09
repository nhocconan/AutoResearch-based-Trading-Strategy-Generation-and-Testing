#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v4
# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Uses Donchian channel breakouts as primary signal, confirmed by 1d volume spike >2.0x 20-period average.
# Choppiness index (14) > 61.8 defines ranging market where we fade breakouts (mean reversion),
# while CHOP < 38.2 defines trending market where we follow breakouts.
# Designed for 20-50 trades/year (80-200 over 4 years) with discrete position sizing to minimize fee drag.
# Works in bull/bear markets: regime filter adapts strategy to market conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v4"
timeframe = "4h"
leverage = 1.0

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First period
    
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:  # Need enough for Donchian(20)
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 4h candle only)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Get 1d HTF data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA and chop
        return np.zeros(n)
    
    # Calculate 1d volume spike (>2.0x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 2.0)
    
    # Calculate 1d Choppiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 4h timeframe (completed daily candle only)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower channel OR chop > 61.8 (strong ranging)
            if close[i] < lower_20_aligned[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper channel OR chop > 61.8 (strong ranging)
            if close[i] > upper_20_aligned[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter based on regime:
            # CHOP < 38.2: trending market -> follow breakouts
            # CHOP > 61.8: ranging market -> fade breakouts (mean reversion)
            if chop_aligned[i] < 38.2:  # Trending - follow breakouts
                # Enter long: price closes above Donchian upper channel with volume spike
                if close[i] > upper_20_aligned[i] and vol_spike_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price closes below Donchian lower channel with volume spike
                elif close[i] < lower_20_aligned[i] and vol_spike_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif chop_aligned[i] > 61.8:  # Ranging - fade breakouts
                # Enter long: price closes below Donchian lower channel with volume spike (mean reversion long)
                if close[i] < lower_20_aligned[i] and vol_spike_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price closes above Donchian upper channel with volume spike (mean reversion short)
                elif close[i] > upper_20_aligned[i] and vol_spike_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals