#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level in low chop regime (CHOP < 38.2) with volume spike
# - Short when price breaks below Camarilla L3 level in low chop regime with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Camarilla levels from 1d provide strong support/resistance, chop filter avoids whipsaws in ranging markets

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Camarilla levels (based on previous day's range)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (use previous completed 1d bar)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index regime filter (CHOP < 38.2 = trending, safe for breakouts)
    # CHOP = 100 * log10(sum(ATR14)/ (max(high-n) - min(low-n))) / log10(n)
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14 = pd.Series(true_ranges).rolling(window=14, min_periods=14).sum()
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_values = chop.fillna(50).values  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    chop_filter = chop_aligned < 38.2  # low chop = trending market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price drops below Camarilla L3 level
            if prices['close'].iloc[i] < camarilla_low_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla H3 level
            if prices['close'].iloc[i] > camarilla_high_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long signal: price breaks above Camarilla H3 in low chop with volume spike
            if (prices['high'].iloc[i] > camarilla_high_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                chop_filter[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short signal: price breaks below Camarilla L3 in low chop with volume spike
            elif (prices['low'].iloc[i] < camarilla_low_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_filter[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals