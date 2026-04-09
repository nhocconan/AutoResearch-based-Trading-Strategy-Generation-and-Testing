#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Camarilla levels from 1d provide precise intraday support/resistance for breakout validation
# Volume spike confirms breakout authenticity; chop regime avoids whipsaws in sideways markets
# Works in bull/bear: Camarilla adapts to volatility, volume confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using prior day's OHLC)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    camarilla_pivot = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:
            continue
        # Use prior day's OHLC
        phigh = df_1d['high'].iloc[i-1]
        plow = df_1d['low'].iloc[i-1]
        pclose = df_1d['close'].iloc[i-1]
        
        pivot = (phigh + plow + pclose) / 3.0
        camarilla_pivot[i] = pivot
        
        # Camarilla levels
        camarilla_h4[i] = phigh + 1.5 * (phigh - plow)
        camarilla_l4[i] = plow - 1.5 * (phigh - plow)
        camarilla_h3[i] = phigh + 1.25 * (phigh - plow)
        camarilla_l3[i] = plow - 1.25 * (phigh - plow)
        camarilla_h2[i] = phigh + 1.0 * (phigh - plow)
        camarilla_l2[i] = plow - 1.0 * (phigh - plow)
        camarilla_h1[i] = phigh + 0.5 * (phigh - plow)
        camarilla_l1[i] = plow - 0.5 * (phigh - plow)
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 12h Donchian breakout (20-period) for entry trigger
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 1d volume spike confirmation (volume > 2x 20-day average)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > 2.0 * vol_ma_20
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 1d choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    # We use CHOP < 50 to avoid whipsaw regimes (more conservative)
    hl_range = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).sum()
    true_range = pd.Series(np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
            np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
        )
    )).rolling(window=14, min_periods=14).sum()
    
    chop = np.full(len(df_1d), np.nan)
    valid = (hl_range.values > 0) & (true_range.values > 0)
    chop[valid] = 100 * np.log10(hl_range.values[valid] / true_range.values[valid]) / np.log10(14)
    chop_filter = chop > 50.0  # Avoid choppy markets (CHOP > 50)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 (strong support break) OR Donchian low break
            if close[i] < camarilla_l3_aligned[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 (strong resistance break) OR Donchian high break
            if close[i] > camarilla_h3_aligned[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout + volume spike + chop filter
            if vol_spike_aligned[i] and chop_filter_aligned[i]:
                # Long entry: price > Camarilla H4 (strong resistance break)
                if close[i] > camarilla_h4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L4 (strong support break)
                elif close[i] < camarilla_l4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals