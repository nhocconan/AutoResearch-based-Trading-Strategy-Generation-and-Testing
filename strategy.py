#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d Camarilla levels (H3/L3) for structure, 1d volume spike (>2.0x 20-period MA) for confirmation.
- Regime filter: 1d Choppiness Index (CHOP) > 61.8 for ranging markets (mean reversion at H3/L3).
- Entry: Long when price breaks above H3 with volume spike AND chop>61.8.
         Short when price breaks below L3 with volume spike AND chop>61.8.
- Exit: Opposite break of H3/L3 or loss of volume confirmation or chop<38.2 (trending regime).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Camarilla levels provide precise support/resistance; chop filter avoids false signals in strong trends.
Works in both bull (breakouts) and bear (mean reversion in ranges) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Extract 1d OHLC
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    # Based on previous day's range
    prev_high = np.roll(df_1d_high, 1)
    prev_low = np.roll(df_1d_low, 1)
    prev_close = np.roll(df_1d_close, 1)
    
    # Set first value to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = prev_close + range_val * 1.1 / 4
    l3 = prev_close - range_val * 1.1 / 4
    h4 = prev_close + range_val * 1.1 / 2
    l4 = prev_close - range_val * 1.1 / 2
    
    # Calculate 1d volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d_volume > (2.0 * vol_ma_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR over n) / (n * (max(high) - min(low)))) / log10(n)
    # We'll use a simplified version: high-low range based
    atr_1d = np.maximum(np.maximum(df_1d_high - df_1d_low, 
                                   np.abs(df_1d_high - np.roll(df_1d_close, 1))),
                        np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    # Handle first bar
    atr_1d[0] = df_1d_high[0] - df_1d_low[0]
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, 
                        100 * np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14), 
                        50)  # neutral when range is zero
    
    # Regime filter: CHOP > 61.8 = ranging (good for mean reversion at H3/L3)
    chop_ranging = chop_raw > 61.8
    # CHOP < 38.2 = trending (avoid mean reversion)
    chop_trending = chop_raw < 38.2
    
    # Align HTF indicators to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need enough bars for 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or
            np.isnan(chop_trending_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike and chop ranging
            if volume_spike_aligned[i] and chop_ranging_aligned[i]:
                # Bullish: price breaks above H3
                if curr_high > h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below L3
                elif curr_low < l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation OR chop becomes trending
            if (curr_low < l3_aligned[i] or 
                not volume_spike_aligned[i] or 
                chop_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation OR chop becomes trending
            if (curr_high > h3_aligned[i] or 
                not volume_spike_aligned[i] or 
                chop_trending_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0