#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Uses 1d Camarilla pivot levels (H4/L4) for breakout signals (long above H4, short below L4)
# - Confirms with 1d volume > 1.5x 20-period average (strong participation)
# - Filters by 1d choppiness index: trade only when CHOP > 61.8 (range) OR CHOP < 38.2 (trend)
# - Exits when price touches opposite Camarilla level (L4 for longs, H4 for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Camarilla levels provide robust support/resistance that adapts to volatility

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR and chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for chop calculation
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Pivot point (standard)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # 1d Camarilla levels (H4/L4)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Align all 1d indicators to 12h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches opposite Camarilla level (L4)
            if low[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches opposite Camarilla level (H4)
            if high[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and regime filter
            if (high[i] >= camarilla_h4_aligned[i] and  # Break above H4
                volume_spike_aligned[i] and         # Volume confirmation
                (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = 1
                signals[i] = 0.25
            elif (low[i] <= camarilla_l4_aligned[i] and   # Break below L4
                  volume_spike_aligned[i] and         # Volume confirmation
                  (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = -1
                signals[i] = -0.25
    
    return signals