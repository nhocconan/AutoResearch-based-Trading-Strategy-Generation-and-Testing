#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high in 1d chop regime (CHOP > 61.8) with volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low in 1d chop regime with volume spike
# - Uses ATR(14) trailing stop: exit when price moves against position by 2.5x ATR
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets ~30 trades/year (120 total over 4 years) to avoid fee drag
# - Chop regime filter ensures trades occur in ranging markets where mean reversion works
# - Volume confirmation ensures institutional participation in breakouts

name = "4h_1d_donchian_chop_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Choppiness Index (CHOP) - measures ranging vs trending
    # CHOP > 61.8 = ranging (good for mean reversion/breakout fade)
    # CHOP < 38.2 = trending (avoid breakouts in strong trends)
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first bar TR
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_14_1d * np.sqrt(14) / (highest_high_14_1d - lowest_low_14_1d)) / np.log10(14)
    chop_1d = np.where((highest_high_14_1d - lowest_low_14_1d) == 0, 50, chop_1d)  # avoid div by zero
    chop_regime_1d = chop_1d > 61.8  # ranging regime
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 4h indicators
    highest_high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    lowest_low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    atr_14 = pd.Series(np.maximum(prices['high'] - prices['low'], 
                                  np.maximum(np.abs(prices['high'] - prices['close'].shift(1)), 
                                             np.abs(prices['low'] - prices['close'].shift(1))))).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_regime_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Donchian breakout above in chop regime with volume spike
            if (prices['close'].iloc[i] > highest_high_20[i] and 
                chop_regime_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = 0.25
            # Short signal: Donchian breakdown below in chop regime with volume spike
            elif (prices['close'].iloc[i] < lowest_low_20[i] and 
                  chop_regime_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                lowest_since_entry = entry_price
                signals[i] = -0.25
        else:  # Have position - manage trailing stop
            if position == 1:  # Long position
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                if prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Short position
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_14[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals