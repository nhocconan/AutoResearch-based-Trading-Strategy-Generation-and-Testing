#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v2
# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and choppiness regime filter.
# Uses Donchian(20) breakouts for trend capture, 1d volume spike to confirm institutional participation,
# and choppiness index to avoid whipsaws in ranging markets. Works in bull/bear markets:
# - Bull: captures breakouts with volume confirmation
# - Bear: choppiness filter prevents false breakouts during consolidation, volume spike identifies panic selling
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v2"
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
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:  # Need enough for Donchian(20)
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over last 20 periods
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 4h candle only)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d HTF data ONCE before loop for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for choppiness calculation
        return np.zeros(n)
    
    # Calculate 1d volume spike (20-period volume average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * log10(highest high - lowest low))) / log10(n)
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    atr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * np.log10(hh_14 - ll_14)
    chop_numerator = np.log10(atr_sum)
    chop = 100 * chop_numerator / chop_denominator
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR choppiness too high (range-bound)
            if (close[i] < lower_4h_aligned[i]) or (chop_aligned[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR choppiness too high
            if (close[i] > upper_4h_aligned[i]) or (chop_aligned[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian upper band, with volume spike, in trending market
            if (close[i] > upper_4h_aligned[i]) and vol_spike_aligned[i] and (chop_aligned[i] < 38.2):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian lower band, with volume spike, in trending market
            elif (close[i] < lower_4h_aligned[i]) and vol_spike_aligned[i] and (chop_aligned[i] < 38.2):
                position = -1
                signals[i] = -0.25
    
    return signals