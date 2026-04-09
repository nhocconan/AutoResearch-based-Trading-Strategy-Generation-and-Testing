#!/usr/bin/env python3
# 4h_donchian_1d_camarilla_volume_v4
# Hypothesis: Further tighten entry by requiring Donchian breakout + volume spike + price closing beyond Camarilla H4/L4 with ATR filter to avoid whipsaws. Target 75-150 trades over 4 years. Works in bull/bear via Donchian breakouts with stronger pivot bias and volatility filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_camarilla_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 4h timeframe (no additional delay needed for price channels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d HTF data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for daily (H4/L4 for stronger direction filter)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (completed daily candle only)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Further increased threshold to reduce trades
    
    # ATR filter to avoid whipsaws (14-period ATR on 4h)
    tr1 = pd.Series(high_4h).rolling(2).apply(lambda x: x[1] - x[0], raw=True).shift(1).fillna(0).values
    tr2 = pd.Series(high_4h).rolling(2).apply(lambda x: abs(x[1] - close_4h[:-1]) if len(close_4h) > 1 else 0, raw=True).shift(1).fillna(0).values
    tr3 = pd.Series(low_4h).rolling(2).apply(lambda x: abs(x[1] - close_4h[:-1]) if len(close_4h) > 1 else 0, raw=True).shift(1).fillna(0).values
    close_4h = df_4h['close'].values
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower band
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper band
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h Donchian upper, above 1d H4, with volume spike and ATR > 0
            if (close[i] > donchian_upper_aligned[i]) and (close[i] > h4_1d_aligned[i]) and vol_spike[i] and (atr_14_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian lower, below 1d L4, with volume spike and ATR > 0
            elif (close[i] < donchian_lower_aligned[i]) and (close[i] < l4_1d_aligned[i]) and vol_spike[i] and (atr_14_aligned[i] > 0):
                position = -1
                signals[i] = -0.25
    
    return signals