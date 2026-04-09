#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Uses Donchian channel breakouts for trend capture, 1d volume spike to confirm institutional participation,
# and choppiness index to avoid whipsaw in ranging markets. Designed for 19-50 trades/year (75-200 over 4 years).
# Works in bull/bear markets: Donchian captures breakouts, volume confirms validity, chop filter prevents false signals in ranges.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v3"
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:  # Need enough for Donchian(20) + buffer
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
    if len(df_1d) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    
    # Calculate 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_ma_20 * 2.0)
    
    # Calculate 1d Choppiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(HH) - Min(LL) over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    hh_ll_diff = hh_14 - ll_14
    
    # Chop = 100 * log10(ATR_sum / hh_ll_diff) / log10(14)
    chop = np.zeros_like(close_1d)
    mask = (atr_sum > 0) & (hh_ll_diff > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / hh_ll_diff[mask]) / np.log10(14)
    
    # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    chop_mask = chop < 38.2  # Only trade in trending regimes
    
    # Align 1d indicators to 4h timeframe (completed daily candle only)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    chop_mask_aligned = align_htf_to_ltf(prices, df_1d, chop_mask)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_mask_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower channel
            if close[i] < lower_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper channel
            if close[i] > upper_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h Donchian upper channel, with volume spike and trending regime
            if (close[i] > upper_20_aligned[i]) and vol_spike_aligned[i] and chop_mask_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian lower channel, with volume spike and trending regime
            elif (close[i] < lower_20_aligned[i]) and vol_spike_aligned[i] and chop_mask_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals